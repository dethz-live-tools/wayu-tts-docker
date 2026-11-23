"""Build-time model preparation tests; downloads are mocked."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import prepare_model


class PrepareModelTests(unittest.TestCase):
    def test_downloads_pinned_checkpoint_with_all_voices(self):
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").touch()
            (model_dir / "model.pth").touch()
            voices = model_dir / "voices"
            voices.mkdir()
            for index in range(12):
                (voices / f"voice_{index}.pt").touch()
            with patch.object(prepare_model, "snapshot_download", return_value=directory) as download:
                self.assertEqual(prepare_model.prepare_model(directory), model_dir)
            download.assert_called_once_with(
                repo_id="wayu-ai/wayu-paxa-tts-edge",
                revision=prepare_model.MODEL_REVISION,
                local_dir=directory,
                allow_patterns=["config.json", "model.pth", "voices/*.pt", "README.md"],
            )
            self.assertEqual(len(prepare_model.MODEL_REVISION), 40)

    def test_rejects_incomplete_checkpoint(self):
        with TemporaryDirectory() as directory:
            with (
                patch.object(prepare_model, "snapshot_download", return_value=directory),
                self.assertRaisesRegex(RuntimeError, "Incomplete Wayu checkpoint"),
            ):
                prepare_model.prepare_model(directory)

    def test_rejects_missing_voices(self):
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").touch()
            (model_dir / "model.pth").touch()
            with (
                patch.object(prepare_model, "snapshot_download", return_value=directory),
                self.assertRaisesRegex(RuntimeError, "12 voices"),
            ):
                prepare_model.prepare_model(directory)


if __name__ == "__main__":
    unittest.main()
