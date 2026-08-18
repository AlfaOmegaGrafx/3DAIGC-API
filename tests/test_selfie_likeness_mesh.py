"""Selfie → MediaPipe face mesh for MeshMonk likeness source."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from core.utils.face_correspondence import selfie_image_to_face_mesh


def test_selfie_image_to_face_mesh_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Selfie image not found"):
        selfie_image_to_face_mesh(tmp_path / "nope.jpg")


def test_selfie_image_to_face_mesh_builds_delaunay(tmp_path: Path):
    img_path = tmp_path / "face.png"
    Image.new("RGB", (64, 64), color=(180, 140, 120)).save(img_path)
    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"fake-model")

    class Lm:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    landmarks = [
        Lm(0.3, 0.3, 0.01),
        Lm(0.7, 0.3, 0.01),
        Lm(0.3, 0.7, 0.02),
        Lm(0.7, 0.7, 0.02),
        Lm(0.5, 0.5, 0.0),
        Lm(0.4, 0.45, 0.01),
        Lm(0.6, 0.45, 0.01),
        Lm(0.5, 0.6, 0.015),
    ]
    mock_result = MagicMock()
    mock_result.face_landmarks = [landmarks]

    mock_landmarker = MagicMock()
    mock_landmarker.__enter__ = MagicMock(return_value=mock_landmarker)
    mock_landmarker.__exit__ = MagicMock(return_value=False)
    mock_landmarker.detect.return_value = mock_result

    with (
        patch(
            "core.utils.face_correspondence._face_landmarker_model_path",
            return_value=model_path,
        ),
        patch(
            "mediapipe.tasks.python.vision.FaceLandmarker.create_from_options",
            return_value=mock_landmarker,
        ),
        patch("mediapipe.Image"),
    ):
        out = selfie_image_to_face_mesh(img_path)

    assert out["source"] == "selfie_mediapipe"
    assert out["vertices"].shape == (8, 3)
    assert out["faces"].ndim == 2 and out["faces"].shape[1] == 3
    assert len(out["faces"]) >= 1
    assert np.isfinite(out["vertices"]).all()
