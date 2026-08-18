"""Unit tests for 1:1 metric scale calibration."""

import pytest

from core.utils.metric_scale import (
    apply_metric_scale_to_manifest,
    euclidean_distance,
    metric_scale_factor,
    metric_scale_from_points,
    resolve_metric_calibration,
)


def test_metric_scale_factor_basic():
    assert metric_scale_factor(true_meters=2.0, recon_length=1.0) == 2.0
    assert metric_scale_factor(true_meters=1.6, recon_length=0.8) == 2.0


def test_metric_scale_rejects_non_positive():
    with pytest.raises(ValueError):
        metric_scale_factor(true_meters=0, recon_length=1)
    with pytest.raises(ValueError):
        metric_scale_factor(true_meters=1, recon_length=0)


def test_two_points_door_width():
    # Door jambs 0.5 units apart in recon, real door 0.9 m
    scale, recon = metric_scale_from_points([0, 0, 0], [0.5, 0, 0], 0.9)
    assert recon == pytest.approx(0.5)
    assert scale == pytest.approx(1.8)


def test_resolve_reference_length():
    resolved = resolve_metric_calibration(
        {"mode": "reference_length", "true_meters": 2.4, "recon_length": 1.2}
    )
    assert resolved["scale"] == pytest.approx(2.0)
    assert resolved["one_to_one"] is True
    assert resolved["units"] == "meters"
    assert resolved["axis"] == "horizontal"


def test_resolve_player_height_default():
    resolved = resolve_metric_calibration({"mode": "player_height", "recon_height": 0.8})
    assert resolved["true_meters"] == pytest.approx(1.6)
    assert resolved["scale"] == pytest.approx(2.0)
    assert resolved["axis"] == "uniform"


def test_apply_metric_scale_to_manifest():
    manifest = {
        "id": "scan-1",
        "version": 1,
        "spawn": {"position": [0, 0, 0]},
        "environment": {
            "type": "gaussian_splat",
            "url": "environment.ply",
            "transform": {"scale": 1},
        },
        "props": [
            {
                "id": "chair",
                "mesh_url": "props/chair.glb",
                "transform": {"position": [1, 0, -2], "scale": 1},
            }
        ],
    }
    # reference_length without axis defaults to horizontal (Office door lock).
    out = apply_metric_scale_to_manifest(
        manifest,
        {"mode": "reference_length", "true_meters": 2.0, "recon_length": 1.0},
    )
    assert out["environment"]["transform"]["scale"] == pytest.approx([2.0, 1.0, 2.0])
    assert out["props"][0]["transform"]["position"] == pytest.approx([2.0, 0.0, -4.0])
    assert out["props"][0]["transform"]["scale"] == pytest.approx([2.0, 1.0, 2.0])
    assert out["spawn"]["player_height"] == 1.6
    assert out["metadata"]["coordinate_units"] == "meters"
    assert out["metadata"]["metric_calibration"]["one_to_one"] is True
    assert out["metadata"]["metric_calibration"]["axis"] == "horizontal"

    # Explicit uniform still stretches XYZ when requested.
    out_u = apply_metric_scale_to_manifest(
        manifest,
        {
            "mode": "reference_length",
            "axis": "uniform",
            "true_meters": 2.0,
            "recon_length": 1.0,
        },
    )
    assert out_u["environment"]["transform"]["scale"] == pytest.approx([2.0, 2.0, 2.0])


def test_resolve_reference_length_defaults_horizontal():
    resolved = resolve_metric_calibration(
        {"mode": "reference_length", "true_meters": 0.762, "recon_length": 0.47}
    )
    assert resolved["axis"] == "horizontal"
    assert resolved["scale"] == pytest.approx(0.762 / 0.47)


def test_apply_horizontal_door_width_keeps_height():
    # 30 in door = 0.762 m; XR jambs = 0.47 m → widen floor plan only
    true_m = 30 * 0.0254
    manifest = {
        "id": "office",
        "environment": {"type": "point_cloud", "url": "environment.ply", "transform": {"scale": 1}},
        "props": [],
    }
    out = apply_metric_scale_to_manifest(
        manifest,
        {
            "mode": "reference_length",
            "axis": "horizontal",
            "true_meters": true_m,
            "recon_length": 0.47,
        },
    )
    sx = true_m / 0.47
    assert out["environment"]["transform"]["scale"] == pytest.approx([sx, 1.0, sx])
    assert out["metadata"]["metric_calibration"]["axis"] == "horizontal"


def test_euclidean_distance():
    assert euclidean_distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)
