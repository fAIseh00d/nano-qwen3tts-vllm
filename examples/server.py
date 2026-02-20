"""FastAPI server for Qwen3-TTS text-to-speech generation.

Env:
  USE_ZMQ=1                  - Use ZMQ (async engine loop + async queue).
  QWEN3_TTS_MODEL_PATH       - Model directory or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-Base).
  HOST, PORT                 - Server bind address.
  OUTPUT_SAMPLE_RATE         - Output sample rate (default: 16000 for voice-backend compatibility).
  VOICES_DIR                 - Directory for voice clone .pkl files (default: ./voices).
  GPU_MEMORY_UTILIZATION     - Fraction of GPU memory to use (default: 0.9, range 0.0-1.0).
  ENFORCE_EAGER              - Disable CUDA graphs for debugging (default: 0).

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
import functools
import io
import json
import logging
import os
import pickle
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
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
    _handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO)

# Suppress verbose ZMQ bridge logging (per-message timings)
logging.getLogger("nano_qwen3tts_vllm.zmq").setLevel(logging.WARNING)

# Lazy imports to avoid loading heavy models at module load
_interface = None
_tokenizer = None
_zmq_bridge = None

# --- Batched decode system (replaces _decode_lock) ---
# Instead of serializing decode with a threading.Lock, requests submit to a
# queue and a background worker batches them into fewer decode calls.
_decode_queue: asyncio.Queue = None
_decode_worker_task: asyncio.Task = None
# Fallback lock for sync warm-up path only (not used in async serving)
_decode_lock = threading.Lock()
# Lock for non-ZMQ generation (sync API is not thread-safe)
_generate_lock = threading.Lock()

# Default speakers for CustomVoice model
DEFAULT_SPEAKERS = [
   
]

# Voice clones directory (configurable via env for docker mount)
VOICES_DIR = Path(os.environ.get("VOICES_DIR", str(Path(__file__).parent / "voices")))
VOICES_DIR.mkdir(exist_ok=True)


def _use_zmq():
    """True if server should use ZMQ (background engine loop + queue-based generate)."""
    return os.environ.get("USE_ZMQ", "1").lower() in ("1", "true", "yes")


def get_interface():
    """Get or initialize the Qwen3TTSInterface (with or without ZMQ based on USE_ZMQ env)."""
    global _interface, _zmq_bridge
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        enforce_eager = os.environ.get("ENFORCE_EAGER", "0").lower() in ("1", "true", "yes")

        # GPU memory: if >1 treat as KV cache GB (model weights added automatically), otherwise as fraction
        # Model weights are ~3.6GB fixed, add buffer for activations = ~4GB overhead
        MODEL_OVERHEAD_GB = 4.0
        gpu_mem_value = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if gpu_mem_value > 1.0:
            # Value is KV cache GB - add model overhead to get total
            kv_cache_gb = gpu_mem_value
            total_needed_gb = kv_cache_gb + MODEL_OVERHEAD_GB
            gpu_mem_util = min(total_needed_gb / total_vram_gb, 0.95)  # Cap at 95%
            logger.info(f"GPU memory: KV={kv_cache_gb:.1f}GB + model={MODEL_OVERHEAD_GB:.1f}GB = {total_needed_gb:.1f}GB / {total_vram_gb:.1f}GB = {gpu_mem_util:.2f} fraction")
        else:
            gpu_mem_util = gpu_mem_value
            kv_cache_gb = (gpu_mem_util * total_vram_gb) - MODEL_OVERHEAD_GB
            logger.info(f"GPU memory: {gpu_mem_util:.0%} of {total_vram_gb:.1f}GB = ~{kv_cache_gb:.1f}GB KV cache")

        logger.info(f"Initializing Qwen3TTSInterface: model={model_path}, gpu_mem={gpu_mem_util:.2f}, enforce_eager={enforce_eager}")

        # Check if it's a local path or HuggingFace model ID
        if os.path.isdir(model_path) or os.path.isfile(model_path):
            # Local path - use regular init
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface(
                    model_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=enforce_eager,
                    gpu_memory_utilization=gpu_mem_util,
                )
            else:
                _interface = Qwen3TTSInterface(
                    model_path=model_path,
                    enforce_eager=enforce_eager,
                    gpu_memory_utilization=gpu_mem_util,
                )
        else:
            # HuggingFace model ID - use from_pretrained
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=enforce_eager,
                    gpu_memory_utilization=gpu_mem_util,
                )
            else:
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    enforce_eager=enforce_eager,
                    gpu_memory_utilization=gpu_mem_util,
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
    """Startup: warm up model, start ZMQ tasks and decode worker. Shutdown: stop all."""
    global _decode_queue, _decode_worker_task

    interface = get_interface()
    get_tokenizer()
    if _use_zmq() and interface.zmq_bridge is not None:
        await interface.start_zmq_tasks()

    # Auto-create voice clones from reference audio files in VOICES_DIR
    create_voice_clones_from_references()

    # Start batched decode worker
    _decode_queue = asyncio.Queue()
    _decode_worker_task = asyncio.create_task(_decode_worker_loop())

    # Warmup: run concurrent requests to compile CUDA kernels for all batch sizes.
    # Without this, the first batch=8 request triggers kernel compilation (~400ms extra).
    # NOTE: Warmup requires a voice clone since Base model doesn't support named speakers.
    voice_clones = get_voice_clones()
    if voice_clones:
        warmup_speaker = voice_clones[0]
        warmup_text = "Hello, this is a warmup test."
        warmup_req = SpeechRequest(text=warmup_text, language="English", speaker=warmup_speaker)

        async def _warmup_one(req):
            try:
                async for _ in generate_speech_stream(req):
                    pass
            except Exception as e:
                logger.warning(f"[warmup] error (non-fatal): {e}")

        # Batch=1 warmup (also warms _do_prep, prefill, predictor, decode)
        logger.info(f"[warmup] batch=1 using voice clone '{warmup_speaker}' ...")
        await _warmup_one(warmup_req)

        # Batch warmup only makes sense with ZMQ (async batching support)
        # Without ZMQ, requests are serialized so batch warmup is skipped
        if _use_zmq():
            logger.info("[warmup] batch=8 ...")
            await asyncio.gather(*[_warmup_one(warmup_req) for _ in range(16)])
        else:
            logger.info("[warmup] skipping batch warmup (non-ZMQ mode is serialized)")
        logger.info("[warmup] done.")
    else:
        logger.info("[warmup] skipped - no voice clones available (Base model requires voice cloning)")

    yield

    # Stop decode worker
    if _decode_queue is not None:
        await _decode_queue.put(None)  # sentinel
    if _decode_worker_task is not None:
        await _decode_worker_task

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
    sample_rate: int = Field(default=None, description="Output sample rate (default: OUTPUT_SAMPLE_RATE)")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/voices")
async def list_voices():
    """
    List all available voices (default speakers + voice clones).
    Returns a dictionary with 'default' and 'clones' keys.
    """
    default_voices = DEFAULT_SPEAKERS
    voice_clones = get_voice_clones()
    
    return {
        "default": default_voices,
        "clones": voice_clones,
        "all": default_voices + voice_clones,
    }


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

async def _decode_worker_loop():
    """Background asyncio task: collects decode requests from _decode_queue,
    micro-batches them, and runs decode in an executor.

    Each queue item is a dict: {"audio_codes": list, "future": asyncio.Future}.
    The worker waits for the first item, then drains any additional queued items
    into a single batched decode call, delivering results to each future.
    """
    decode_start = time.time()
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    while True:
        # Wait for first request
        item = await _decode_queue.get()
        if item is None:
            break
        batch = [item]

        # Drain any queued requests (micro-batch)
        while not _decode_queue.empty():
            try:
                extra = _decode_queue.get_nowait()
                if extra is None:
                    break
                batch.append(extra)
            except asyncio.QueueEmpty:
                break

        # Batched decode: combine all into one call
        all_inputs = [{"audio_codes": req["audio_codes"]} for req in batch]

        def _do_decode(inputs=all_inputs):
            return tokenizer.decode(inputs)

        try:
            decode_start = time.time()
            wav_results, sr = await loop.run_in_executor(None, _do_decode)
            decode_latency = (time.time() - decode_start) * 1000
            logger.info(f"[decoder] batch_size={len(batch)} latency={decode_latency:.2f}ms")
            for req, wav in zip(batch, wav_results):
                wav_resampled = _resample(wav, sr, OUTPUT_SAMPLE_RATE)
                pcm16 = _float_to_pcm16(wav_resampled)
                if not req["future"].done():
                    req["future"].set_result((pcm16, OUTPUT_SAMPLE_RATE))
        except Exception as e:
            for req in batch:
                if not req["future"].done():
                    req["future"].set_exception(e)
                    
        logger.info(f"[decoder] total latency: {time.time() - decode_start:.2f}s")


async def _decode_batched(audio_codes: list) -> tuple[np.ndarray, int]:
    """Submit a decode request to the batched worker and await the result."""
    future = asyncio.get_event_loop().create_future()
    await _decode_queue.put({"audio_codes": audio_codes, "future": future})
    return await future


def _decode_inline(audio_codes: list, output_sr: int = None) -> tuple[np.ndarray, int]:
    """Decode directly on the calling thread (no executor).

    Used for FIRST-chunk decode to avoid the ~130ms scheduling delay caused by
    run_in_executor callback delivery competing with GPU prefills on the event loop.
    Single-code CUDA-graph decode takes ~15ms, acceptable for inline use.
    """
    output_sr = output_sr or OUTPUT_SAMPLE_RATE
    tokenizer = get_tokenizer()
    with torch.inference_mode():
        wav_list, sr = tokenizer.chunked_decode([{"audio_codes": audio_codes}])
    wav_resampled = _resample(wav_list[0], sr, output_sr)
    return _float_to_pcm16(wav_resampled), output_sr


def get_voice_clones():
    """Get list of saved voice clone names from the voices directory."""
    if not VOICES_DIR.exists():
        return []
    voice_files = list(VOICES_DIR.glob("*.pkl"))
    voice_names = [f.stem for f in voice_files]
    return sorted(voice_names)


def create_voice_clones_from_references():
    """
    Auto-create voice clone .pkl files from reference audio in VOICES_DIR.
    Looks for .wav/.mp3/.flac files and creates .pkl if not exists.
    Optional .txt file with same name provides reference text for better quality.
    """
    if not VOICES_DIR.exists():
        return

    interface = get_interface()
    audio_extensions = (".wav", ".mp3", ".flac", ".ogg")

    for audio_file in VOICES_DIR.iterdir():
        if not audio_file.suffix.lower() in audio_extensions:
            continue

        pkl_path = VOICES_DIR / f"{audio_file.stem}.pkl"
        if pkl_path.exists():
            continue  # Already created

        # Check for optional transcript
        txt_path = VOICES_DIR / f"{audio_file.stem}.txt"
        ref_text = None
        if txt_path.exists():
            ref_text = txt_path.read_text().strip()

        try:
            logger.info(f"Creating voice clone from {audio_file.name} (ref_text={bool(ref_text)})")

            # Interface expects file path (str), not bytes
            voice_clone_prompt = interface.create_voice_clone_prompt(
                ref_audio=str(audio_file),
                ref_text=ref_text,
                x_vector_only_mode=not bool(ref_text),
            )

            with open(pkl_path, 'wb') as f:
                pickle.dump(voice_clone_prompt, f)

            logger.info(f"Created voice clone: {pkl_path.name}")
        except Exception as e:
            logger.error(f"Failed to create voice clone from {audio_file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())


def is_voice_clone(speaker: str) -> bool:
    """Check if speaker is a voice clone (exists in voices directory)."""
    return (VOICES_DIR / f"{speaker}.pkl").exists()


@functools.lru_cache(maxsize=128)
def load_voice_clone_prompt(speaker: str):
    """Load voice clone prompt from pickle file."""
    voice_path = VOICES_DIR / f"{speaker}.pkl"
    if not voice_path.exists():
        raise ValueError(f"Voice clone '{speaker}' not found")
    with open(voice_path, 'rb') as f:
        return pickle.load(f)


async def generate_voice_clone_codes(interface, text: str, language: str, voice_clone_prompt, loop=None):
    """
    Generate voice clone codes, using async API if ZMQ is enabled, sync API otherwise.
    Yields audio codes one at a time for streaming.
    """
    if _use_zmq():
        # Async streaming API (preferred with ZMQ)
        async for code in interface.generate_voice_clone_async(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        ):
            yield code
    else:
        # Sync API wrapped in executor for non-ZMQ mode
        # Use lock to serialize - sync API is not thread-safe
        loop = loop or asyncio.get_event_loop()
        def _generate_sync():
            with _generate_lock:
                return list(interface.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                ))
        codes = await loop.run_in_executor(None, _generate_sync)
        for code in codes:
            yield code




async def generate_speech_stream(request: SpeechRequest, output_sr: int = None):
    """
    Streaming decode: first chunk decoded inline for minimal latency,
    subsequent chunks use producer/consumer with batched decode worker.

    Key insight: putting the first code through codes_queue → consumer adds
    ~130ms scheduling delay (consumer competes with talker/predictor for event
    loop time). By awaiting the first code directly via __anext__() and decoding
    inline, we eliminate that delay entirely.

    CANCELLATION SAFETY: when the client disconnects mid-stream, we must:
      1. Cancel the producer task (stops consuming from gen immediately)
      2. Close gen (triggers interface.py's finally → clear_request)
    Without this, a ghost sequence stays in scheduler.running and causes
    200ms batch_wait timeouts on every subsequent talker step.
    """
    gen = None
    producer_task = None
    try:
        interface = get_interface()
        tokenizer = get_tokenizer()
        loop = asyncio.get_event_loop()
        start_time = time.time()

        # --- Determine voice clone prompt (required for Base model) ---
        text = "..." + request.text
        if is_voice_clone(request.speaker):
            voice_clone_prompt = load_voice_clone_prompt(request.speaker)
        else:
            # Base model requires voice cloning - fall back to first available
            available_clones = get_voice_clones()
            if available_clones:
                fallback = available_clones[0]
                logger.info(f"Speaker '{request.speaker}' not found, using fallback: {fallback}")
                voice_clone_prompt = load_voice_clone_prompt(fallback)
            else:
                raise ValueError("No voice clone available. Create one via /v1/audio/speech/clone or WebSocket with ref_audio.")

        # Create generator (works with both ZMQ and non-ZMQ modes)
        gen = generate_voice_clone_codes(interface, text, request.language, voice_clone_prompt, loop)

        # --- FIRST CHUNK: collect 2 codes, decode inline, yield immediately ---
        # Waiting for 2 codes gives a longer initial audio segment for smoother
        # playback start, at the cost of one extra decode cycle (~50ms).
        FIRST_CHUNK_CODES = 2
        first_codes = []
        async for audio_code in gen:
            first_codes.append(audio_code)
            if len(first_codes) >= FIRST_CHUNK_CODES:
                break
        if not first_codes:
            return  # generator produced nothing

        t_first_code = time.time()
        pcm16_first, _ = _decode_inline(first_codes, output_sr)
        t_first_decoded = time.time()
        logger.info(
            f"[stream] first chunk codes={len(first_codes)} "
            f"latency: {(t_first_code - start_time)*1000:.1f}ms "
            f"decode: {(t_first_decoded - t_first_code)*1000:.1f}ms "
            f"total: {(t_first_decoded - start_time)*1000:.1f}ms"
        )
        prev_len_24k = len(pcm16_first)
        yield pcm16_first.tobytes()

        # --- SUBSEQUENT CHUNKS: producer task + batched decode worker ---
        codes_queue: asyncio.Queue[list | None] = asyncio.Queue()  # unbounded

        async def producer() -> None:
            audio_codes = list(first_codes)  # first codes already yielded
            last_chunk_time = t_first_code
            try:
                async for audio_code in gen:
                    current_time = time.time()
                    inner_latency = current_time - last_chunk_time
                    logger.debug(f"[producer] inner chunk latency: {inner_latency*1000:.2f}ms")
                    last_chunk_time = current_time

                    audio_codes.append(audio_code)
                    if len(audio_codes) % 4 == 0:
                        await codes_queue.put(list(audio_codes))
                # final partial batch
                if audio_codes and len(audio_codes) % 4 != 0:
                    await codes_queue.put(list(audio_codes))
            finally:
                await codes_queue.put(None)  # sentinel

        producer_task = asyncio.create_task(producer())

        try:
            while True:
                item = await codes_queue.get()
                if item is None:
                    break
                if _decode_queue is not None:
                    pcm16, _ = await _decode_batched(item)
                else:
                    pcm16, _ = await loop.run_in_executor(
                        None,
                        lambda c=item, sr=output_sr: _decode_batch(tokenizer, c, sr),
                    )
                chunk = pcm16[prev_len_24k:].tobytes()
                prev_len_24k = len(pcm16)
                if chunk:
                    yield chunk
        finally:
            # Cancel producer immediately -- don't wait for it to finish naturally.
            # Without cancel(), a disconnected client leaves the producer running
            # for seconds, keeping the ghost sequence in scheduler.running.
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass
    except StopAsyncIteration:
        # Generator produced no codes at all
        pass
    except Exception as e:
        logger.error(f"[generate_speech_stream] Error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # ALWAYS close the interface generator to trigger its cleanup
        # (clear_request in interface.py's finally block removes the sequence
        # from scheduler.running, preventing ghost sequence / batch_wait timeouts).
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                pass  # may already be closed from normal EOS

@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def generate_speech(request: SpeechRequest):
    """
    Generate speech from text.
    Returns raw PCM 16-bit mono at OUTPUT_SAMPLE_RATE (default 16kHz).
    Uses generate_custom_voice_async (requires USE_ZMQ=1).
    """
    output_sr = request.sample_rate or OUTPUT_SAMPLE_RATE
    try:
        return StreamingResponse(
            generate_speech_stream(request, output_sr),
            media_type="audio/L16",
            headers={"Sample-Rate": str(output_sr)},
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
        "voices": "GET /voices (list all available voices)",
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
            """Generate audio codes using async API (required for ZMQ)."""
            nonlocal audio_codes
            gen = None
            try:
                # Determine which voice clone prompt to use
                prompt = None
                if use_voice_cloning and voice_clone_prompt:
                    logger.info("Using inline voice cloning mode")
                    prompt = voice_clone_prompt
                elif is_voice_clone(speaker):
                    logger.info(f"Using saved voice clone: {speaker}")
                    prompt = load_voice_clone_prompt(speaker)
                else:
                    available = get_voice_clones()
                    if available:
                        fallback = available[0]
                        logger.info(f"No voice provided, using fallback: {fallback}")
                        prompt = load_voice_clone_prompt(fallback)
                    else:
                        raise ValueError("No voice clone available. Provide ref_audio or create a voice clone first.")

                # Generate codes (works with both ZMQ and non-ZMQ modes)
                text = "..." + text_buffer
                gen = generate_voice_clone_codes(interface, text, language, prompt, loop)

                # Stream codes and batch for decode
                async for audio_code in gen:
                    audio_codes.append(audio_code)
                    if len(audio_codes) % 4 == 0:
                        await codes_queue.put(list(audio_codes))

                # Final batch
                if audio_codes and len(audio_codes) % 4 != 0:
                    await codes_queue.put(list(audio_codes))

                logger.info(f"Generated {len(audio_codes)} codes")
            except Exception as e:
                logger.error(f"Error in audio generation: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                await codes_queue.put(None)  # Sentinel
                if gen is not None:
                    try:
                        await gen.aclose()
                    except Exception:
                        pass

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


class VoiceCloneCreateRequest(BaseModel):
    """Request body for creating/saving a voice clone."""
    name: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$',
                      description="Voice clone name (alphanumeric, underscore, dash only)")
    audio: str = Field(..., description="Base64-encoded reference audio (WAV/MP3/FLAC)")
    transcript: str = Field(default="", description="Transcript of the audio (improves clone quality)")


class VoiceCloneResponse(BaseModel):
    """Response for voice clone operations."""
    success: bool
    name: str
    message: str


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


@app.post("/v1/voice-clone/create", response_model=VoiceCloneResponse)
async def create_voice_clone(request: VoiceCloneCreateRequest):
    """
    Create and save a voice clone from reference audio.

    The voice clone is saved as a .pkl file in VOICES_DIR and can be used
    immediately with /v1/audio/speech by specifying the voice name as speaker.
    """
    interface = get_interface()
    loop = asyncio.get_event_loop()

    try:
        # Check if name already exists
        pkl_path = VOICES_DIR / f"{request.name}.pkl"
        if pkl_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Voice clone '{request.name}' already exists. Use PUT to update or DELETE first."
            )

        # Decode audio
        try:
            audio_bytes = base64.b64decode(request.audio)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e}")

        logger.info(f"Creating voice clone '{request.name}': {len(audio_bytes)} bytes audio, transcript={bool(request.transcript)}")

        # Create voice clone prompt
        # Interface expects file path, so save to temp file first
        import tempfile
        def _create_prompt():
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                return interface.create_voice_clone_prompt(
                    ref_audio=tmp_path,
                    ref_text=request.transcript if request.transcript else None,
                    x_vector_only_mode=not bool(request.transcript),
                )
            finally:
                os.unlink(tmp_path)  # Clean up temp file

        voice_clone_prompt = await loop.run_in_executor(None, _create_prompt)

        # Save to file
        with open(pkl_path, 'wb') as f:
            pickle.dump(voice_clone_prompt, f)

        # Clear cache so new voice is immediately available
        load_voice_clone_prompt.cache_clear()

        logger.info(f"Voice clone '{request.name}' created successfully")

        return VoiceCloneResponse(
            success=True,
            name=request.name,
            message=f"Voice clone '{request.name}' created successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create voice clone: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/voice-clone/{name}", response_model=VoiceCloneResponse)
async def delete_voice_clone(name: str):
    """Delete a voice clone by name."""
    pkl_path = VOICES_DIR / f"{name}.pkl"

    if not pkl_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice clone '{name}' not found")

    try:
        pkl_path.unlink()
        load_voice_clone_prompt.cache_clear()

        logger.info(f"Voice clone '{name}' deleted")

        return VoiceCloneResponse(
            success=True,
            name=name,
            message=f"Voice clone '{name}' deleted successfully"
        )
    except Exception as e:
        logger.error(f"Failed to delete voice clone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/voice-clone/{name}")
async def get_voice_clone_info(name: str):
    """Get info about a specific voice clone."""
    pkl_path = VOICES_DIR / f"{name}.pkl"

    if not pkl_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice clone '{name}' not found")

    stat = pkl_path.stat()

    return {
        "name": name,
        "exists": True,
        "size_bytes": stat.st_size,
        "created_at": stat.st_ctime,
        "modified_at": stat.st_mtime,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    if _use_zmq():
        logger.info("Starting Qwen3-TTS API with ZMQ (async engine loop).")
    uvicorn.run(app, host=host, port=port)
