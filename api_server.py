import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("uvicorn.error")
MODEL_ID = "wayu-ai/wayu-paxa-tts-edge"
inference_lock = Lock()


def load_model():
    from wayu_tts import ThaiTTS

    source = os.environ.get("WAYU_MODEL_PATH") or MODEL_ID
    if source != MODEL_ID:
        model_dir = Path(source)
        if not all((model_dir / name).exists() for name in ("config.json", "model.pth", "voices")):
            raise RuntimeError(
                f"Pretrained model files are missing from {source}. Rebuild the Docker image "
                "or set WAYU_MODEL_PATH to a complete model directory."
            )
    logger.info("Loading %s from %s on CPU", MODEL_ID, source)
    return ThaiTTS.from_pretrained(source, device="cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tts = await run_in_threadpool(load_model)
    logger.info("Wayu-Paxa-TTS-Edge ready")
    try:
        yield
    finally:
        app.state.tts = None


app = FastAPI(title="Wayu-Paxa-TTS-Edge API", lifespan=lifespan)
app.state.tts = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Latency-Ms"],
)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: Literal[
        "f_teen_bright", "f_young_bright", "f_young_clear", "f_young_warm",
        "f_mid_clear", "f_mid_warm", "f_elderly_soft", "f_elderly_low",
        "m_teen_bright", "m_young_clear", "m_mid_warm", "m_elderly_deep",
    ] = "f_young_clear"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank")
        return value


def get_model():
    model = app.state.tts
    if model is None:
        raise HTTPException(status_code=503, detail="Wayu model is not ready.")
    return model


def synthesize_wayu(req: TTSRequest) -> io.BytesIO:
    model = get_model()
    # The model caches voices and the frontend lazily; serialize access to shared state.
    with inference_lock:
        audio = model(req.text, voice=req.voice, speed=req.speed)
    if audio.size == 0:
        raise HTTPException(status_code=422, detail="Text produced no speakable audio.")
    buffer = io.BytesIO()
    sf.write(buffer, audio, model.sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer


@app.get("/health")
async def health():
    model = get_model()
    return {"status": "ok", "backend": "wayu", "model": MODEL_ID, "device": model.device}


@app.get("/api/voices")
async def voices():
    return {"voices": list(get_model().voices)}


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    start = time.perf_counter()
    try:
        audio = await run_in_threadpool(synthesize_wayu, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Wayu synthesis failed")
        raise HTTPException(status_code=500, detail="Speech synthesis failed.") from exc

    return StreamingResponse(
        audio,
        media_type="audio/wav",
        headers={"X-Latency-Ms": str(int((time.perf_counter() - start) * 1000))},
    )


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>TTS API Playground</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #0f172a; color: #f8fafc; }
        textarea, select { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: white; font-size: 16px; margin-top: 8px; }
        textarea { height: 140px; }
        button { margin-top: 12px; padding: 12px 24px; background: #06b6d4; color: #0f172a; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 16px; }
        button:disabled { opacity: .5; cursor: wait; }
        audio { width: 100%; margin-top: 20px; }
        a { color: #22d3ee; }
    </style>
</head>
<body>
    <h1>🎙️ TTS API Playground</h1>
    <p>พิมพ์คำ ➔ สร้างเสียงด้วย Wayu-Paxa-TTS-Edge บน CPU</p>
    <p>เสียงสังเคราะห์ AI · โมเดล CC-BY-NC-4.0 สำหรับการใช้งานที่ไม่ใช่เชิงพาณิชย์</p>
    <form id="tts-form">
        <label for="txt">ข้อความ</label>
        <textarea id="txt" required maxlength="5000">สวัสดีครับ ทดสอบยิงข้อความเข้า API แล้วได้เสียงพูดออกมาทันที</textarea>
        <label for="voice">เสียง Wayu</label>
        <select id="voice">
            <option value="f_teen_bright">หญิงวัยรุ่น — bright</option>
            <option value="f_young_bright">หญิงวัยหนุ่มสาว — bright</option>
            <option value="f_young_clear" selected>หญิงวัยหนุ่มสาว — clear</option>
            <option value="f_young_warm">หญิงวัยหนุ่มสาว — warm</option>
            <option value="f_mid_clear">หญิงวัยกลางคน — clear</option>
            <option value="f_mid_warm">หญิงวัยกลางคน — warm</option>
            <option value="f_elderly_soft">หญิงสูงวัย — soft</option>
            <option value="f_elderly_low">หญิงสูงวัย — low</option>
            <option value="m_teen_bright">ชายวัยรุ่น — bright</option>
            <option value="m_young_clear">ชายวัยหนุ่มสาว — clear</option>
            <option value="m_mid_warm">ชายวัยกลางคน — warm</option>
            <option value="m_elderly_deep">ชายสูงวัย — deep</option>
        </select>
        <button id="generate" type="submit">🚀 ส่งเข้า API (/api/tts)</button>
    </form>
    <p id="status" role="status"></p>
    <audio id="player" controls></audio>
    <p><a href="/docs">API documentation</a></p>
    <script>
        let audioUrl;
        document.getElementById('tts-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const status = document.getElementById('status');
            const button = document.getElementById('generate');
            button.disabled = true;
            status.textContent = 'กำลังเรียก POST /api/tts ...';
            try {
                const res = await fetch('/api/tts', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        text: document.getElementById('txt').value,
                        voice: document.getElementById('voice').value,
                        speed: 1.0
                    })
                });
                if (!res.ok) throw new Error('API error: ' + res.status + ' ' + await res.text());
                const blob = await res.blob();
                const player = document.getElementById('player');
                if (audioUrl) URL.revokeObjectURL(audioUrl);
                audioUrl = URL.createObjectURL(blob);
                player.src = audioUrl;
                status.textContent = '✅ ได้รับไฟล์เสียงสำเร็จ! (' + res.headers.get('X-Latency-Ms') + ' ms)';
                try { await player.play(); }
                catch { status.textContent += ' กด Play เพื่อฟังเสียง'; }
            } catch (error) {
                status.textContent = '❌ ' + error.message;
            } finally {
                button.disabled = false;
            }
        });
    </script>
</body>
</html>
"""
