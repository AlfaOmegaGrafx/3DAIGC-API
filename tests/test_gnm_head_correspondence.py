"""Unit tests for GNM head helpers + face correspondence."""
from __future__ import annotations

import numpy as np
import pytest

from core.utils.face_correspondence import (
    apply_morph_preserving_delta,
    deform_template_neutral,
    load_template_head_neutral,
    rigid_scale_align,
    sanitize_warp_delta,
)
from core.utils.gnm_head import (
    normalize_ethnicity,
    normalize_gender,
)


def test_normalize_gender_ethnicity():
    assert normalize_gender("Male") == "male"
    assert normalize_gender("woman") == "female"
    assert normalize_gender("") is None
    assert normalize_ethnicity("Asian") == "asian"
    assert normalize_ethnicity("middle-eastern") == "middle_eastern"
    assert normalize_ethnicity("caucasian") == "white"
    assert normalize_ethnicity("") is None


def test_morph_preserving_delta_invariant():
    neutral = np.zeros((10, 3), dtype=np.float32)
    morph = np.zeros((10, 3), dtype=np.float32)
    morph[:, 0] = 1.0
    delta = np.zeros((10, 3), dtype=np.float32)
    delta[:, 1] = 0.5
    new_n, new_m = apply_morph_preserving_delta(neutral, [morph], delta)
    assert np.allclose(new_n[:, 1], 0.5)
    # Expression offset along X preserved relative to new neutral
    assert np.allclose(new_m[0] - new_n, morph - neutral)


def test_deform_template_neutral_alpha_zero():
    tgt = np.random.default_rng(0).normal(size=(40, 3)).astype(np.float32)
    src = tgt + 1.0
    delta = deform_template_neutral(src, tgt, alpha=0.0)
    assert np.allclose(delta, 0.0)


def test_sanitize_warp_delta_clamps_explosion():
    rng = np.random.default_rng(2)
    # Unit-ish head bbox (~1.0 diagonal)
    template = rng.normal(size=(60, 3)).astype(np.float32) * 0.2
    # Huge pull that previously exploded eyes/teeth visually
    delta = np.zeros_like(template)
    delta[:, 1] = 5.0
    safe, info = sanitize_warp_delta(delta, template, max_frac=0.22, rms_frac=0.07)
    assert info["clamped"] is True
    assert float(np.linalg.norm(safe, axis=1).max()) <= info["diag"] * 0.22 + 1e-5


def test_sanitize_warp_delta_rejects_extreme():
    template = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    delta = np.ones_like(template) * 100.0
    safe, info = sanitize_warp_delta(
        delta, template, max_frac=0.22, rms_frac=0.07, min_keep_scale=0.15
    )
    assert info["rejected"] is True
    assert np.allclose(safe, 0.0)


def test_sanitize_warp_delta_reject_if_clamped():
    rng = np.random.default_rng(3)
    template = rng.normal(size=(40, 3)).astype(np.float32) * 0.15
    delta = np.zeros_like(template)
    delta[:, 1] = 0.8  # exceeds soft clamp but would have been scale-clamped
    safe, info = sanitize_warp_delta(
        delta, template, max_frac=0.12, rms_frac=0.04, reject_if_clamped=True
    )
    assert info["rejected"] is True
    assert np.allclose(safe, 0.0)


def test_deform_template_neutral_rbf_explicit():
    rng = np.random.default_rng(1)
    tgt = rng.normal(size=(48, 3)).astype(np.float32)
    src = tgt * np.array([1.1, 0.95, 1.05], dtype=np.float32)
    delta = deform_template_neutral(src, tgt, method="rbf", alpha=1.0)
    assert delta.shape == tgt.shape
    assert float(np.sqrt((delta**2).mean())) > 0.0


@pytest.mark.slow
def test_meshmonk_warp_icosphere():
    pytest.importorskip("meshmonk")
    import trimesh

    from core.utils.face_correspondence import meshmonk_warp

    base = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    src_v = np.asarray(base.vertices, dtype=np.float32) * np.array(
        [1.08, 0.94, 1.03], dtype=np.float32
    )
    faces = np.asarray(base.faces, dtype=np.int32)
    warped = meshmonk_warp(
        np.asarray(base.vertices, dtype=np.float32),
        src_v,
        template_faces=faces,
        source_faces=faces,
        num_iterations=20,
        num_pyramid_layers=2,
    )
    assert warped.shape == base.vertices.shape
    # Should move toward the scaled source more than staying put.
    rms_before = float(np.sqrt(((base.vertices - src_v) ** 2).mean()))
    rms_after = float(np.sqrt(((warped - src_v) ** 2).mean()))
    assert rms_after < rms_before * 0.85


def test_rigid_scale_align_centers():
    src = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
    tgt = np.array([[10, 10, 10], [12, 10, 10], [10, 12, 10]], dtype=np.float32)
    aligned = rigid_scale_align(src, tgt)
    assert aligned.shape == src.shape
    assert np.allclose(aligned.mean(axis=0), tgt.mean(axis=0), atol=0.5)


def test_load_template_head_neutral():
    from pathlib import Path

    vrm = Path(__file__).resolve().parents[1] / "assets/example_autorig/template.vrm"
    if not vrm.is_file():
        pytest.skip("template.vrm missing")
    info = load_template_head_neutral(vrm)
    assert info["vertices"].shape[1] == 3
    assert info["vertices"].shape[0] > 100
    assert info["morph_count"] >= 8
    assert "head" in info["mesh_name"].lower()
    assert info.get("faces") is not None
    assert info["faces"].shape[1] == 3
    assert len(info["faces"]) > 100


@pytest.mark.slow
def test_gnm_sample_identity_smoke():
    pytest.importorskip("gnm")
    from core.utils.gnm_head import generate_identity_mesh

    mesh = generate_identity_mesh(gender="female", ethnicity="asian", seed=7)
    assert mesh["vertices"].shape[1] == 3
    assert mesh["vertices"].shape[0] > 1000
    assert mesh["identity"].ndim == 1
