# Qwen3-TTS Server with WebSocket Streaming Support
#
# Features:
# - WebSocket endpoint for streaming text input from LLM
# - Voice cloning support via reference audio
# - HTTP endpoint for voice cloning

FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install flash-attn from pre-built wheel
RUN pip install --break-system-packages --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.10-cp312-cp312-linux_x86_64.whl

# Copy and install nano-qwen3tts-vllm
WORKDIR /app
COPY pyproject.toml .
COPY nano-qwen3tts-vllm ./nano-qwen3tts-vllm
COPY examples ./examples
RUN pip install --break-system-packages --no-cache-dir -e .

# Environment
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    USE_ZMQ=1 \
    OUTPUT_SAMPLE_RATE=16000 \
    HOST=0.0.0.0 \
    PORT=8002

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "examples/server.py"]
