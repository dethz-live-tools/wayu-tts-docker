"""Wayu-Paxa API contract tests; no real model is imported or downloaded."""

import builtins
from contextlib import ExitStack
import importlib
import importlib.util
import io
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

import numpy as np
from fastapi.concurrency import run_in_threadpool
from fastapi.testclient import TestClient
from pydantic import ValidationError


MODEL_ID = "wayu-ai/wayu-paxa-tts-edge"
# Voice IDs published in the Wayu-Paxa-TTS-Edge model card.
VOICES = [
    "f_teen_bright",
    "f_young_bright",
    "f_young_clear",
    "f_young_warm",
    "f_mid_clear",
    "f_mid_warm",
    "f_elderly_soft",
    "f_elderly_low",
    "m_teen_bright",
    "m_young_clear",
    "m_mid_warm",
    "m_elderly_deep",
]
SECRET_ERROR = "private model path C:/secret/weights; token=do-not-expose"


class FakeTTS:
    sample_rate = 24000
    device = "cpu"

    def __init__(self):
        self.voices = list(VOICES)
        self.synthesize = Mock(
            side_effect=lambda *args, **kwargs: np.zeros(240, dtype=np.float32)
        )

    def __call__(self, text, *, voice, speed):
        return self.synthesize(text, voice=voice, speed=speed)


def fake_wayu_module():
    module = ModuleType("wayu_tts")
    module.ThaiTTS = Mock()
    module.ThaiTTS.from_pretrained.side_effect = AssertionError(
        "A test attempted to load an unmocked model"
    )
    return module


# Guard even the initial API import against accidental eager model loading.
with patch.dict(sys.modules, {"wayu_tts": fake_wayu_module()}):
    api_server = importlib.import_module("api_server")


class ModelLoadingTests(unittest.TestCase):
    def test_api_import_does_not_import_or_load_wayu(self):
        module = fake_wayu_module()
        spec = importlib.util.spec_from_file_location(
            "_api_server_import_test", api_server.__file__
        )
        isolated_api = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, {"wayu_tts": module}),
            patch("builtins.__import__", wraps=builtins.__import__) as imports,
            patch("importlib.import_module", wraps=importlib.import_module) as dynamic,
        ):
            spec.loader.exec_module(isolated_api)
        imported_names = [call.args[0] for call in imports.call_args_list]
        imported_names += [call.args[0] for call in dynamic.call_args_list]
        self.assertFalse(
            any(name == "wayu_tts" or name.startswith("wayu_tts.")
                for name in imported_names),
            "wayu_tts must only be imported by load_model(), not at API import time",
        )
        module.ThaiTTS.from_pretrained.assert_not_called()

    def test_load_model_uses_thai_tts_model_id_and_cpu(self):
        module = fake_wayu_module()
        model = FakeTTS()
        module.ThaiTTS.from_pretrained.side_effect = None
        module.ThaiTTS.from_pretrained.return_value = model
        with (
            patch.dict(sys.modules, {"wayu_tts": module}),
            patch.dict(os.environ, {"WAYU_MODEL_PATH": ""}),
        ):
            loaded = api_server.load_model()
        self.assertEqual(api_server.MODEL_ID, MODEL_ID)
        self.assertIs(loaded, model)
        module.ThaiTTS.from_pretrained.assert_called_once_with(MODEL_ID, device="cpu")

    def test_load_model_uses_bundled_weights(self):
        module = fake_wayu_module()
        module.ThaiTTS.from_pretrained.side_effect = None
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").touch()
            (model_dir / "model.pth").touch()
            (model_dir / "voices").mkdir()
            with (
                patch.dict(sys.modules, {"wayu_tts": module}),
                patch.dict(os.environ, {"WAYU_MODEL_PATH": directory}),
            ):
                api_server.load_model()
            module.ThaiTTS.from_pretrained.assert_called_once_with(directory, device="cpu")

    def test_missing_bundled_weights_fail_without_hub_fallback(self):
        module = fake_wayu_module()
        with TemporaryDirectory() as directory:
            with (
                patch.dict(sys.modules, {"wayu_tts": module}),
                patch.dict(os.environ, {"WAYU_MODEL_PATH": directory}),
                self.assertRaisesRegex(RuntimeError, "Rebuild the Docker image"),
            ):
                api_server.load_model()
        module.ThaiTTS.from_pretrained.assert_not_called()

    def test_startup_loads_in_threadpool_and_shutdown_clears_model(self):
        model = FakeTTS()
        with (
            patch.object(api_server.app.state, "tts", None, create=True),
            patch.object(api_server, "load_model", return_value=model) as loader,
            patch.object(
                api_server, "run_in_threadpool", AsyncMock(wraps=run_in_threadpool)
            ) as threadpool,
        ):
            with TestClient(api_server.app):
                self.assertIs(api_server.app.state.tts, model)
                loader.assert_called_once_with()
                threadpool.assert_awaited_once_with(loader)
            self.assertIsNone(api_server.app.state.tts)

    def test_startup_load_failure_propagates_without_fallback(self):
        failure = RuntimeError(SECRET_ERROR)
        with (
            patch.object(api_server.app.state, "tts", None, create=True),
            patch.object(api_server, "load_model", side_effect=failure) as loader,
        ):
            with self.assertRaises(RuntimeError) as raised:
                with TestClient(api_server.app):
                    self.fail("Startup must not succeed after model loading fails")
            self.assertIs(raised.exception, failure)
            loader.assert_called_once_with()
            self.assertIsNone(api_server.app.state.tts)


class ReadyAPITests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.model = FakeTTS()
        stack.enter_context(patch.object(api_server.app.state, "tts", None, create=True))
        self.loader = stack.enter_context(
            patch.object(api_server, "load_model", return_value=self.model)
        )
        self.client = stack.enter_context(
            TestClient(api_server.app, raise_server_exceptions=False)
        )

    def assert_wav(self, response, sample_rate=24000):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"].split(";")[0], "audio/wav")
        with wave.open(io.BytesIO(response.content), "rb") as audio:
            self.assertEqual(audio.getframerate(), sample_rate)
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getcomptype(), "NONE")
            self.assertEqual(audio.getnframes(), 240)
            self.assertEqual(audio.readframes(240), bytes(240 * 2))

    def test_health_identifies_loaded_cpu_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ok", "backend": "wayu", "model": MODEL_ID, "device": "cpu"
        })

    def test_voices_returns_loaded_model_voices(self):
        # A tuple also verifies that the endpoint serializes list(tts.voices).
        self.model.voices = tuple(reversed(VOICES))
        response = self.client.get("/api/voices")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"voices": list(self.model.voices)})
        self.model.synthesize.assert_not_called()

    def test_tts_strips_text_and_uses_defaults(self):
        response = self.client.post("/api/tts", json={"text": " \nสวัสดีครับ\t "})
        self.assert_wav(response)
        self.model.synthesize.assert_called_once_with(
            "สวัสดีครับ", voice="f_young_clear", speed=1.0
        )

    def test_tts_accepts_every_model_voice_and_passes_speed(self):
        for voice in VOICES:
            with self.subTest(voice=voice):
                self.model.synthesize.reset_mock()
                response = self.client.post(
                    "/api/tts", json={"text": "Hello", "voice": voice, "speed": 1.25}
                )
                self.assert_wav(response)
                self.model.synthesize.assert_called_once_with(
                    "Hello", voice=voice, speed=1.25
                )

    def test_synthesize_wayu_returns_rewound_wav_buffer(self):
        request = api_server.TTSRequest(text="Hello", voice="m_mid_warm", speed=0.75)
        audio = api_server.synthesize_wayu(request)
        self.assertIsInstance(audio, io.BytesIO)
        self.assertEqual(audio.tell(), 0)
        with wave.open(audio, "rb") as wav:
            self.assertEqual(wav.getframerate(), 24000)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), 240)
        self.model.synthesize.assert_called_once_with(
            "Hello", voice="m_mid_warm", speed=0.75
        )

    def test_tts_uses_model_sample_rate(self):
        self.model.sample_rate = 16000
        self.assert_wav(
            self.client.post("/api/tts", json={"text": "Hello"}), sample_rate=16000
        )

    def test_tts_dispatches_synthesize_wayu_through_threadpool(self):
        with patch.object(
            api_server, "run_in_threadpool", AsyncMock(wraps=run_in_threadpool)
        ) as threadpool:
            response = self.client.post("/api/tts", json={"text": " Hello "})
        self.assert_wav(response)
        threadpool.assert_awaited_once()
        args = threadpool.await_args.args
        self.assertIs(args[0], api_server.synthesize_wayu)
        self.assertIsInstance(args[1], api_server.TTSRequest)
        self.assertEqual(args[1].text, "Hello")

    def test_text_and_speed_boundaries_are_accepted(self):
        for text, speed in [("a", 0.5), ("ก" * 5000, 2.0)]:
            with self.subTest(length=len(text), speed=speed):
                self.assert_wav(self.client.post(
                    "/api/tts", json={"text": text, "speed": speed}
                ))

    def test_invalid_requests_return_422_without_synthesis(self):
        payloads = [
            {}, {"text": None}, {"text": ""}, {"text": " \n\t "},
            {"text": "ก" * 5001},
            {"text": "Hello", "voice": "unknown"},
            {"text": "Hello", "voice": "th-TH-PremwadeeNeural"},
            {"text": "Hello", "voice": None},
            {"text": "Hello", "speed": 0.49},
            {"text": "Hello", "speed": 2.01},
            {"text": "Hello", "speed": "fast"},
        ]
        for payload in payloads:
            with self.subTest(payload=repr(payload)[:100]):
                response = self.client.post("/api/tts", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
        self.model.synthesize.assert_not_called()

    def test_voice_field_is_literal_of_all_twelve_voices(self):
        schema = api_server.TTSRequest.model_json_schema()
        self.assertCountEqual(schema["properties"]["voice"]["enum"], VOICES)
        self.assertEqual(schema["properties"]["voice"]["default"], "f_young_clear")

    def test_request_model_rejects_blank_text_directly(self):
        with self.assertRaises(ValidationError):
            api_server.TTSRequest(text=" \n\t ")

    def test_empty_model_audio_returns_422(self):
        self.model.synthesize.side_effect = None
        self.model.synthesize.return_value = np.zeros(0, dtype=np.float32)
        response = self.client.post("/api/tts", json={"text": "Hello"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("detail", response.json())

    def test_wav_encoding_failure_returns_generic_500(self):
        with patch.object(api_server.sf, "write", side_effect=RuntimeError(SECRET_ERROR)) as write:
            response = self.client.post("/api/tts", json={"text": "Hello"})
        write.assert_called_once()
        self.assertEqual(response.status_code, 500, response.text)
        self.assertNotIn(SECRET_ERROR, response.text)
        self.assertNotIn("do-not-expose", response.text)
        self.assertNotIn("Traceback", response.text)

    def test_model_failure_returns_generic_500_without_leaking_exception(self):
        self.model.synthesize.side_effect = RuntimeError(SECRET_ERROR)
        response = self.client.post("/api/tts", json={"text": "Hello"})
        self.assertEqual(response.status_code, 500, response.text)
        self.assertNotIn(SECRET_ERROR, response.text)
        self.assertNotIn("RuntimeError", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("do-not-expose", response.text)


class UnreadyAPITests(unittest.TestCase):
    def test_unready_endpoints_return_503_without_attempting_model_load(self):
        with (
            patch.object(api_server.app.state, "tts", None, create=True),
            patch.object(api_server, "load_model") as loader,
        ):
            # Deliberately do not enter TestClient: lifespan must not run here.
            client = TestClient(api_server.app, raise_server_exceptions=False)
            self.addCleanup(client.close)
            for method, path, kwargs in [
                ("GET", "/health", {}),
                ("GET", "/api/voices", {}),
                ("POST", "/api/tts", {"json": {"text": "Hello"}}),
            ]:
                with self.subTest(path=path):
                    response = client.request(method, path, **kwargs)
                    self.assertEqual(response.status_code, 503, response.text)
                    self.assertNotIn("Traceback", response.text)
            loader.assert_not_called()
            self.assertIsNone(api_server.app.state.tts)


if __name__ == "__main__":
    unittest.main()
