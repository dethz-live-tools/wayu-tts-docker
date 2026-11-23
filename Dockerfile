FROM ghcr.io/astral-sh/uv:0.11.18 AS uv
FROM python:3.11-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    WAYU_MODEL_PATH=/models/wayu-paxa-tts-edge \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git espeak-ng libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface /models/wayu-paxa-tts-edge \
    && chown -R app:app /home/app/.cache /models
COPY --chown=app:app api_server.py prepare_model.py ./
USER app

RUN python prepare_model.py --output /models/wayu-paxa-tts-edge

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Fail the build if the bundled checkpoint or Thai/English frontend cannot synthesize.
RUN python -c "from api_server import load_model; tts = load_model(); audio = tts('สวัสดีครับ Hello world', voice='m_young_clear'); assert audio.size > 0, 'Model returned no audio'"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uv", "run", "--no-sync", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
