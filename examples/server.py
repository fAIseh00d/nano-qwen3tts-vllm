"""FastAPI server for Qwen3-TTS text-to-speech generation.

Env:
  USE_ZMQ=1              - Use ZMQ (async engine loop + async queue).
  QWEN3_TTS_MODEL_PATH   - Model directory or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
  HOST, PORT             - Server bind address.
  OUTPUT_SAMPLE_RATE     - Output sample rate (default: 16000 for voice-backend compatibility).

WebSocket streaming endpoint for LLM token input:
  ws://host:port/v1/audio/speech/stream
  Protocol:
    1. Client sends config JSON: {ref_audio?, ref_text?, sample_rate?, language?, speaker?}
    2. Client sends text tokens as strings
    3. Server sends audio chunks as binary (PCM16)
    4. Client sends {"done": true} to signal end
    5. Server sends {"done": true} after all audio sent
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import threading
from contextlib import asynccontextmanager
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Internal decode rate (codec output)
CODEC_SAMPLE_RATE = 24000
# Output sample rate (configurable, default 16kHz for voice-backend compatibility)
OUTPUT_SAMPLE_RATE = int(os.environ.get("OUTPUT_SAMPLE_RATE", "16000"))
# Legacy alias
TARGET_SAMPLE_RATE = CODEC_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Ensure log messages appear on console (works when run as uvicorn server:app or python server.py)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO)

# Lazy imports to avoid loading heavy models at module load
_interface = None
_tokenizer = None
_zmq_bridge = None
_decode_lock = threading.Lock()


def _use_zmq():
    """True if server should use ZMQ (background engine loop + queue-based generate)."""
    return os.environ.get("USE_ZMQ", "1").lower() in ("1", "true", "yes")


def get_interface():
    """Get or initialize the Qwen3TTSInterface (with or without ZMQ based on USE_ZMQ env)."""
    global _interface, _zmq_bridge
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        # Check if it's a local path or HuggingFace model ID
        if os.path.isdir(model_path) or os.path.isfile(model_path):
            # Local path - use regular init
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                # Auto-find port if default is in use
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface(
                    model_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=False,
                )
            else:
                _interface = Qwen3TTSInterface(model_path=model_path)
        else:
            # HuggingFace model ID - use from_pretrained
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                # Auto-find port if default is in use
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=False,
                )
            else:
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    enforce_eager=False,
                )
    return _interface


def get_tokenizer():
    """Get or initialize the Qwen3TTSTokenizer for decoding audio codes."""
    global _tokenizer
    if _tokenizer is None:
        from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph

        _tokenizer = SpeechTokenizerCUDAGraph(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device="cuda:0",
        )
        
    return _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up model and start ZMQ tasks when USE_ZMQ. Shutdown: stop ZMQ tasks and close bridge."""
    interface = get_interface()
    get_tokenizer()
    if _use_zmq() and interface.zmq_bridge is not None:
        await interface.start_zmq_tasks()
        
    generate_speech_stream(SpeechRequest(text="Hello, this is a test.", language="English", speaker=""))
    yield
    if _use_zmq() and _interface is not None and _interface.zmq_bridge is not None:
        await _interface.stop_zmq_tasks()
        if _zmq_bridge is not None:
            _zmq_bridge.close()


app = FastAPI(
    title="Qwen3-TTS API",
    description="Text-to-speech generation using Qwen3-TTS with vLLM-style optimizations",
    version="0.1.0",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    """Request body for speech generation."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="English", description="Language of the text")
    speaker: str = Field(default="", description="Speaker name (empty for voice cloning mode)")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def _float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 PCM."""
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16)


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform to target sample rate if needed."""
    if orig_sr == target_sr:
        return wav
    n_orig = len(wav)
    n_new = int(round(n_orig * target_sr / orig_sr))
    if n_new == 0:
        return wav
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)


def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample waveform to 24 kHz if needed (legacy function)."""
    return _resample(wav, orig_sr, CODEC_SAMPLE_RATE)


def _decode_batch(tokenizer, audio_codes: list, output_sr: int = None) -> tuple[np.ndarray, int]:
    """Decode cumulative audio_codes to PCM16. Returns (pcm16, sample_rate)."""
    output_sr = output_sr or OUTPUT_SAMPLE_RATE
    with _decode_lock:
        wav_list, sr = tokenizer.decode([{"audio_codes": audio_codes}])
    wav = wav_list[0]
    # First resample to codec rate if needed, then to output rate
    wav_resampled = _resample(wav, sr, output_sr)
    return _float_to_pcm16(wav_resampled), output_sr


async def generate_speech_stream(request: SpeechRequest):
    """
    Streaming decode: producer (generation) and consumer (decode) run concurrently.
    Uses single asyncio.Queue + run_in_executor — no decode thread, no call_soon_threadsafe.
    When consumer awaits decode in executor, event loop runs producer (overlap).
    """
    interface = get_interface()
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    codes_queue: asyncio.Queue[list | None] = asyncio.Queue(maxsize=2)  # backpressure
    async def producer() -> None:
        audio_codes = []
        first_chunk_time = None
        last_chunk_time = None
        try:
            async for audio_code in interface.generate_custom_voice_async(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
            ):
                current_time = time.time()
                if first_chunk_time is None:
                    first_chunk_time = current_time
                if last_chunk_time is not None:
                    inner_latency = current_time - last_chunk_time
                    print(f"[producer] inner chunk latency: {inner_latency*1000:.2f}ms")
                last_chunk_time = current_time
                
                audio_codes.append(audio_code)
                if len(audio_codes) % 4 == 0:  # decode every 4 chunks
                    await codes_queue.put(list(audio_codes))
            
            if first_chunk_time is not None:
                first_chunk_latency = last_chunk_time - first_chunk_time
                print(f"[producer] first chunk latency: {first_chunk_latency*1000:.2f}ms")
            
            # final batch if not already sent (e.g. 13 chunks: sent at 12, need 13)
            if audio_codes and len(audio_codes) % 4 != 0:
                await codes_queue.put(list(audio_codes))
        finally:
            await codes_queue.put(None)  # sentinel

    producer_task = asyncio.create_task(producer())
    prev_len_24k = 0

    try:
        while True:
            item = await codes_queue.get()
            if item is None:
                break
            # run_in_executor: decode in thread pool; event loop runs producer meanwhile
            pcm16, _ = await loop.run_in_executor(
                None,
                lambda c=item: _decode_batch(tokenizer, c),
            )
            chunk = pcm16[prev_len_24k:].tobytes()
            prev_len_24k = len(pcm16)
            if chunk:
                yield chunk
    finally:
        await producer_task


@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def generate_speech(request: SpeechRequest):
    """
    Generate speech from text.
    Returns raw PCM 16-bit mono at 24 kHz (audio/L16).
    Uses generate_custom_voice_async (requires USE_ZMQ=1).
    """
    try:
        return StreamingResponse(
            generate_speech_stream(request),
            media_type="audio/L16",
            headers={"Sample-Rate": str(TARGET_SAMPLE_RATE)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Qwen3-TTS API",
        "docs": "/docs",
        "health": "/health",
        "speech": "POST /v1/audio/speech (PCM16, 24 kHz mono)",
        "speech_stream": "WS /v1/audio/speech/stream (streaming text input)",
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "zmq": _use_zmq(),
    }


# ============================================================================
# WebSocket Streaming Endpoint - Accepts streaming text input from LLM
# ============================================================================

@app.websocket("/v1/audio/speech/stream")
async def speech_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming text input from LLM.

    Protocol:
    1. Client connects
    2. Client sends config JSON: {ref_audio?, ref_text?, sample_rate?, language?, speaker?}
       - ref_audio: Base64-encoded reference audio for voice cloning
       - ref_text: Transcript of reference audio (required for ICL mode)
       - sample_rate: Output sample rate (default: OUTPUT_SAMPLE_RATE)
       - language: Language hint (default: "Auto")
       - speaker: Speaker name for non-cloning mode (default: "Vivian")
    3. Client sends text tokens as plain strings
    4. Server sends audio chunks as binary (PCM16)
    5. Client sends {"done": true} to signal end of text
    6. Server sends {"done": true} JSON after all audio is sent
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    interface = get_interface()
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()

    try:
        # 1. Receive configuration
        config_msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        config = json.loads(config_msg)
        logger.info(f"WebSocket config received: {list(config.keys())}")

        ref_audio_b64 = config.get("ref_audio")
        ref_text = config.get("ref_text", "")
        output_sr = config.get("sample_rate", OUTPUT_SAMPLE_RATE)
        language = config.get("language", "Auto")
        speaker = config.get("speaker", "")

        # 2. Create voice clone prompt if reference audio provided
        voice_clone_prompt = None
        use_voice_cloning = False

        if ref_audio_b64:
            try:
                # Decode base64 audio
                audio_bytes = base64.b64decode(ref_audio_b64)
                logger.info(f"Decoding reference audio: {len(audio_bytes)} bytes")

                # Create voice clone prompt (runs in executor to not block)
                def _create_voice_clone_prompt():
                    return interface.create_voice_clone_prompt(
                        ref_audio=audio_bytes,
                        ref_text=ref_text if ref_text else None,
                        x_vector_only_mode=not bool(ref_text),  # Use x-vector only if no ref_text
                    )

                voice_clone_prompt = await loop.run_in_executor(None, _create_voice_clone_prompt)
                use_voice_cloning = True
                logger.info(f"Voice clone prompt created (x_vector_only={not bool(ref_text)})")
            except Exception as e:
                logger.error(f"Failed to create voice clone prompt: {e}")
                # Fall back to speaker mode

        # 3. Collect text tokens
        text_buffer = ""
        done_receiving = asyncio.Event()

        async def receive_tokens():
            """Receive text tokens from WebSocket until done signal."""
            nonlocal text_buffer
            try:
                while True:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=60.0)

                    if msg["type"] == "websocket.disconnect":
                        logger.info("WebSocket client disconnected during token reception")
                        break

                    if "text" in msg:
                        data = msg["text"]
                        try:
                            # Check for done signal (must be a dict with "done" key)
                            parsed = json.loads(data)
                            if isinstance(parsed, dict) and parsed.get("done"):
                                logger.info(f"Received done signal. Total text: {len(text_buffer)} chars")
                                break
                            else:
                                # Valid JSON but not a done signal - treat as text token
                                text_buffer += data
                        except json.JSONDecodeError:
                            # Regular text token
                            text_buffer += data

            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tokens")
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            finally:
                done_receiving.set()

        # Start receiving tokens in background
        receive_task = asyncio.create_task(receive_tokens())

        # 4. Wait for all tokens (we need complete text for generation)
        # Note: True streaming would require model changes to accept incremental text
        await done_receiving.wait()

        if not text_buffer.strip():
            logger.warning("No text received, closing")
            await websocket.send_json({"done": True, "error": "No text received"})
            await websocket.close()
            return

        logger.info(f"Generating audio for text ({len(text_buffer)} chars): {text_buffer[:100]}...")

        # 5. Generate audio
        generation_start = time.time()
        audio_codes = []
        codes_queue: asyncio.Queue[list | None] = asyncio.Queue(maxsize=2)

        async def producer():
            """Generate audio codes."""
            nonlocal audio_codes
            try:
                if use_voice_cloning and voice_clone_prompt:
                    # Voice cloning mode (sync generator, run in executor)
                    logger.info("Using voice cloning mode")
                    def _generate_voice_clone():
                        return list(interface.generate_voice_clone(
                            text=text_buffer,
                            language=language,
                            voice_clone_prompt=voice_clone_prompt,
                            ref_text=ref_text if ref_text else None,
                        ))

                    audio_codes = await loop.run_in_executor(None, _generate_voice_clone)
                    logger.info(f"Voice cloning generated {len(audio_codes)} codes")
                    # Send codes in batches for incremental decoding
                    for i in range(0, len(audio_codes), 4):
                        batch = audio_codes[:i+4]
                        await codes_queue.put(list(batch))
                else:
                    # Speaker mode (async generator)
                    logger.info(f"Using speaker mode (speaker={speaker or 'default'})")
                    async for audio_code in interface.generate_custom_voice_async(
                        text=text_buffer,
                        language=language,
                        speaker=speaker,
                    ):
                        audio_codes.append(audio_code)
                        if len(audio_codes) % 4 == 0:
                            await codes_queue.put(list(audio_codes))

                    # Final batch
                    if audio_codes and len(audio_codes) % 4 != 0:
                        await codes_queue.put(list(audio_codes))
                    logger.info(f"Speaker mode generated {len(audio_codes)} codes")
            except Exception as e:
                logger.error(f"Error in audio generation: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                await codes_queue.put(None)  # Sentinel

        producer_task = asyncio.create_task(producer())
        prev_samples = 0

        # 6. Stream audio chunks back
        try:
            while True:
                item = await codes_queue.get()
                if item is None:
                    break

                # Decode in executor
                pcm16, sr = await loop.run_in_executor(
                    None,
                    lambda c=item: _decode_batch(tokenizer, c, output_sr),
                )

                # Send only new samples
                new_chunk = pcm16[prev_samples:].tobytes()
                prev_samples = len(pcm16)

                if new_chunk:
                    await websocket.send_bytes(new_chunk)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during audio streaming")
        finally:
            await producer_task

        # 7. Send completion signal
        generation_time = time.time() - generation_start
        try:
            await websocket.send_json({"done": True})
            logger.info(f"WebSocket streaming complete: {len(audio_codes)} codes, {prev_samples} samples, {generation_time*1000:.0f}ms generation")
        except Exception:
            pass  # Client may have disconnected

    except asyncio.TimeoutError:
        logger.error("WebSocket timeout waiting for config")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# HTTP Voice Clone Endpoint (alternative to WebSocket)
# ============================================================================

class VoiceCloneRequest(BaseModel):
    """Request body for voice clone speech generation."""
    text: str = Field(..., min_length=1, description="Text to synthesize")
    ref_audio: str = Field(..., description="Base64-encoded reference audio")
    ref_text: str = Field(default="", description="Transcript of reference audio (optional, improves quality)")
    language: str = Field(default="Auto", description="Language of the text")


@app.post("/v1/audio/speech/clone", response_class=StreamingResponse)
async def generate_speech_clone(request: VoiceCloneRequest):
    """
    Generate speech with voice cloning.
    Returns raw PCM 16-bit mono at OUTPUT_SAMPLE_RATE (default 16kHz).
    """
    interface = get_interface()
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()

    try:
        # Decode reference audio
        audio_bytes = base64.b64decode(request.ref_audio)

        # Create voice clone prompt
        def _create_prompt():
            return interface.create_voice_clone_prompt(
                ref_audio=audio_bytes,
                ref_text=request.ref_text if request.ref_text else None,
                x_vector_only_mode=not bool(request.ref_text),
            )

        voice_clone_prompt = await loop.run_in_executor(None, _create_prompt)

        # Generate audio
        async def generate_stream():
            def _generate():
                return list(interface.generate_voice_clone(
                    text=request.text,
                    language=request.language,
                    voice_clone_prompt=voice_clone_prompt,
                    ref_text=request.ref_text if request.ref_text else None,
                ))

            audio_codes = await loop.run_in_executor(None, _generate)

            # Decode and stream in batches
            prev_samples = 0
            for i in range(4, len(audio_codes) + 1, 4):
                batch = audio_codes[:i]
                pcm16, _ = await loop.run_in_executor(
                    None,
                    lambda c=batch: _decode_batch(tokenizer, c, OUTPUT_SAMPLE_RATE),
                )
                chunk = pcm16[prev_samples:].tobytes()
                prev_samples = len(pcm16)
                if chunk:
                    yield chunk

            # Final batch
            if len(audio_codes) % 4 != 0:
                pcm16, _ = await loop.run_in_executor(
                    None,
                    lambda: _decode_batch(tokenizer, audio_codes, OUTPUT_SAMPLE_RATE),
                )
                chunk = pcm16[prev_samples:].tobytes()
                if chunk:
                    yield chunk

        return StreamingResponse(
            generate_stream(),
            media_type="audio/L16",
            headers={"Sample-Rate": str(OUTPUT_SAMPLE_RATE)},
        )

    except Exception as e:
        logger.error(f"Voice clone error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    if _use_zmq():
        logger.info("Starting Qwen3-TTS API with ZMQ (async engine loop).")
    uvicorn.run(app, host=host, port=port)
