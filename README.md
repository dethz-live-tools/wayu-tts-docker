# Wayu-Paxa-TTS-Edge API

Local Thai–English text-to-speech using [wayu-ai/wayu-paxa-tts-edge](https://huggingface.co/wayu-ai/wayu-paxa-tts-edge): 82M parameters, 12 fixed voices, 24 kHz WAV output. Uses the official inference package's `ThaiTTS` API on CPU, with no Microsoft/edge-tts fallback.

## License

**Model weights: CC-BY-NC-4.0 — attribution required, commercial use not permitted.** The inference library code is separately licensed Apache-2.0. Review the [model card](https://huggingface.co/wayu-ai/wayu-paxa-tts-edge) and [acceptable-use terms](https://www.wayuresearch.org/terms#acceptable-use) before use. This is a research model with no maintenance or availability guarantee. Output is synthetic speech, not a recording of a real person.

## Docker Compose

Compose automatically reads `.env` in the project directory. For a fresh checkout, copy `.env.example` to `.env` first.

```dotenv
TTS_HOST=127.0.0.1
TTS_PORT=8000
```

`TTS_HOST` controls the host interface and `TTS_PORT` controls the published port; the container always listens on port 8000. Keep the default host for local-only access. Setting `TTS_HOST=0.0.0.0` exposes the unauthenticated API to the network. `.env` is ignored by Git and excluded from the Docker build context. No new environment variables or API keys are required for the model.

```sh
docker compose up --build -d
docker compose logs -f tts-api
```

Open http://localhost:8000 for the playground or http://localhost:8000/docs for the API documentation (with the default port). Rerun `up --build -d` after changing the configuration or code.

The image includes the **pretrained weights and all 12 voices**. No training is needed. Docker downloads the pinned Hugging Face checkpoint during **build**, not API startup, so the first build needs Internet access and may take several minutes. `prepare_model.py` verifies the downloaded files; the Dockerfile then runs a real Thai–English synthesis smoke test. Failed downloads or inference fail the build.

At runtime, `WAYU_MODEL_PATH=/models/wayu-paxa-tts-edge` selects the bundled model directory. Hugging Face and Transformers are set to offline mode, and the English spaCy model is preinstalled. There is no model-cache volume to populate or accidentally mask the bundled files. The API accepts requests after loading the local weights and does not send text to a remote speech service. A missing local checkpoint fails startup rather than switching providers.

If upgrading from the old Compose configuration, rebuild and recreate the service:

```sh
docker compose up --build --force-recreate -d
```

The old `model-cache` volume is no longer used; existing volumes are not deleted automatically.

The CPU image includes PyTorch CPU wheels, the official inference package pinned to a Git commit, `espeak-ng` for English fallback, and libsndfile for WAV encoding. It runs as a non-root user. The `/health` readiness check allows a two-minute initial startup period. CPU synthesis requests are serialized to protect the model's shared state; long requests may queue.

Stop the service:

```sh
docker compose down
```

Stopping or recreating the container does not remove the pretrained weights: they are part of the image. The model checkpoint is pinned to revision `8196688df56a8d08ccfa186505343adff412ab6b` in `prepare_model.py`.

## Docker without Compose

```sh
docker build -t tts-api .
docker run --rm --name tts-api -p 127.0.0.1:8000:8000 tts-api
```

uv is available inside the image. Dependencies are installed using `uv sync --locked` from `uv.lock`.

## API

```sh
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"สวัสดีครับ วันนี้เรียก Grab ไปทำงาน","voice":"m_young_clear","speed":1.0}' \
  --output speech.wav
```

- `text`: non-empty text, up to 5,000 characters (API limit).
- `voice`: defaults to `f_young_clear`; must be one of the model's 12 fixed voices. Old Microsoft voice names are no longer accepted.
- `speed`: from `0.5` to `2.0`, default `1.0`. Multiplies the voice's calibrated speaking rate.
- Response: PCM-16 WAV at the model's sample rate (24 kHz); `X-Latency-Ms` includes queue and synthesis time.
- `GET /api/voices`: available voice IDs.
- `GET /health`: model readiness, model ID and device.

| Female voices | Male voices |
| --- | --- |
| `f_teen_bright` | `m_teen_bright` |
| `f_young_bright` | `m_young_clear` |
| `f_young_clear` | `m_mid_warm` |
| `f_young_warm` | `m_elderly_deep` |
| `f_mid_clear` | |
| `f_mid_warm` | |
| `f_elderly_soft` | |
| `f_elderly_low` | |

## Local development

Requires uv and Python 3.11. This version is intentional: tltk requires scikit-learn 1.2.x, whose prebuilt wheels do not support Python 3.12. NumPy is constrained below 2 for binary compatibility. Install Git and, on Debian/Ubuntu, `espeak-ng`, `libsndfile1` and `libgomp1`. Local installations on other operating systems may need equivalent native libraries.

```sh
uv sync --locked
uv run uvicorn api_server:app --reload --port 8000
```

Outside Docker, the API defaults to downloading/loading the Hugging Face model. To mirror Docker's local-file loading, run `uv run python prepare_model.py --output models/wayu-paxa-tts-edge` and set `WAYU_MODEL_PATH` to that directory before starting the API. The Dockerfile sets this automatically; no `.env` changes are required.

After changing dependencies, run `uv lock` and rebuild the image.

Run the mocked API regression tests (no model download):

```sh
uv run --group dev python -m unittest -v
```

CORS permits all origins. The API has no authentication or rate limiting; keep it local or add those protections before exposing it publicly. This setup targets CPU inference; GPU execution requires a separate CUDA-compatible dependency and container configuration.
