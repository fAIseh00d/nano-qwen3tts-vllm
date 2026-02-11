# Fork Requirements for nano-qwen3tts-vllm

This document describes the modifications needed to fork [tsdocode/nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for WebSocket streaming text input support.

## Why Fork?

The original nano-qwen3tts-vllm supports:
- **Output streaming**: Audio chunks are streamed back to the client
- **Input**: Complete text must be provided upfront (HTTP POST)

For true token-level streaming from LLM to TTS, we need:
- **Input streaming**: Accept text tokens via WebSocket as they arrive from LLM
- **Bidirectional**: Send audio chunks back while still receiving text tokens

## Modifications Required

### 1. Add WebSocket Endpoint to `examples/server.py`

Add the following endpoint (~100-150 lines):

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import base64

@app.websocket("/v1/audio/speech/stream")
async def speech_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming text input from LLM.

    Protocol:
    1. Client connects
    2. Client sends config JSON: {ref_audio, ref_text, sample_rate, language}
    3. Client sends text tokens as strings
    4. Server sends audio chunks as binary
    5. Client sends {"done": true} to signal end
    6. Server sends {"done": true} after all audio sent
    """
    await websocket.accept()

    try:
        # 1. Receive configuration
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)

        ref_audio_b64 = config.get("ref_audio")
        ref_text = config.get("ref_text", "")
        sample_rate = config.get("sample_rate", 16000)
        language = config.get("language", "auto")

        # Decode reference audio if provided
        ref_audio = None
        if ref_audio_b64:
            ref_audio = base64.b64decode(ref_audio_b64)

        # 2. Create voice clone prompt (if ref audio provided)
        voice_prompt = None
        if ref_audio:
            voice_prompt = interface.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text
            )

        # 3. Create async generator for incoming tokens
        token_queue = asyncio.Queue()
        done_receiving = asyncio.Event()

        async def token_generator():
            """Yield tokens as they arrive via WebSocket."""
            while True:
                try:
                    token = await asyncio.wait_for(
                        token_queue.get(),
                        timeout=30.0  # 30 second timeout between tokens
                    )
                    if token is None:  # Sentinel for end
                        break
                    yield token
                except asyncio.TimeoutError:
                    break

        async def receive_tokens():
            """Receive tokens from WebSocket and put in queue."""
            try:
                while True:
                    msg = await websocket.receive()

                    if msg["type"] == "websocket.disconnect":
                        break

                    if "text" in msg:
                        data = msg["text"]
                        try:
                            # Check for done signal
                            parsed = json.loads(data)
                            if parsed.get("done"):
                                break
                        except json.JSONDecodeError:
                            # Regular text token
                            await token_queue.put(data)

            except WebSocketDisconnect:
                pass
            finally:
                await token_queue.put(None)  # Signal end
                done_receiving.set()

        # 4. Start receiving tokens in background
        receive_task = asyncio.create_task(receive_tokens())

        # 5. Generate and stream audio
        try:
            if voice_prompt:
                # Voice cloning mode
                async for audio_chunk in interface.generate_voice_clone_streaming(
                    text_iterator=token_generator(),
                    voice_clone_prompt=voice_prompt,
                    language=language
                ):
                    await websocket.send_bytes(audio_chunk)
            else:
                # Default voice mode
                async for audio_chunk in interface.generate_streaming(
                    text_iterator=token_generator(),
                    language=language
                ):
                    await websocket.send_bytes(audio_chunk)

        except Exception as e:
            logger.error(f"Error generating audio: {e}")

        # 6. Wait for receive task and send completion
        await receive_task
        await websocket.send_json({"done": True})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
```

### 2. Add Streaming Generation Method to Interface

If not already present, add these methods to the TTS interface:

```python
async def generate_streaming(
    self,
    text_iterator: AsyncIterator[str],
    language: str = "auto"
) -> AsyncIterator[bytes]:
    """
    Generate audio from streaming text input.

    Args:
        text_iterator: Async iterator yielding text tokens
        language: Language hint

    Yields:
        PCM audio chunks (16-bit, 16kHz, mono)
    """
    # Accumulate tokens and generate as they arrive
    # The exact implementation depends on the model's streaming capability
    pass

async def generate_voice_clone_streaming(
    self,
    text_iterator: AsyncIterator[str],
    voice_clone_prompt: Any,
    language: str = "auto"
) -> AsyncIterator[bytes]:
    """
    Generate audio with voice cloning from streaming text input.
    """
    pass
```

### 3. Add Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME}
```

## Testing the Fork

### 1. Test HTTP Endpoint (existing)

```bash
curl -X POST http://localhost:8002/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello world", "sample_rate": 16000}' \
    --output test.pcm

# Play with: aplay -f S16_LE -r 16000 -c 1 test.pcm
```

### 2. Test WebSocket Endpoint (new)

```python
import asyncio
import aiohttp
import json

async def test_websocket():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect("ws://localhost:8002/v1/audio/speech/stream") as ws:
            # Send config
            await ws.send_json({"sample_rate": 16000})

            # Send tokens
            for token in ["Hello", " ", "world", "!"]:
                await ws.send_str(token)
                await asyncio.sleep(0.05)  # Simulate LLM token delay

            # Signal done
            await ws.send_json({"done": True})

            # Receive audio
            audio_data = b""
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    audio_data += msg.data
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    if json.loads(msg.data).get("done"):
                        break

            print(f"Received {len(audio_data)} bytes of audio")

            # Save audio
            with open("test_ws.pcm", "wb") as f:
                f.write(audio_data)

asyncio.run(test_websocket())
```

## Alternative: Use ZMQ (Already in nano-qwen3tts-vllm)

The original repo has ZMQ support for async generation. This could potentially be adapted for WebSocket streaming without major changes:

```bash
# Enable ZMQ mode
export USE_ZMQ=1
python examples/server.py
```

However, ZMQ is typically used for internal async processing, not client-facing APIs. WebSocket is more suitable for our use case as it's HTTP-compatible and works through firewalls/proxies.

## GPU Memory Requirements

- Qwen3-TTS-12Hz-1.7B-Base: ~4GB VRAM
- With voice cloning prompt: +~0.5GB
- Total: ~4.5GB VRAM

This leaves room for sharing the GPU with vLLM (Qwen3-8B-AWQ at ~10GB).
