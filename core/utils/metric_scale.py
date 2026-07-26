"""
Metric (1:1 meter) scale helpers for environment scans.

LingBot-Map / monocular recon produce unitless (or arbitrary) coordinates.
To anchor a physical-replica metaverse we apply a uniform scale so a known
real-world length matches the same length in the reconstruction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        raise ValueError("Points must be at least 3D (x, y, z)")
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def metric_scale_factor(
    *,
    true_meters: float,
    recon_length: float,
) -> float:
    """
    Return the uniform scale S such that recon_length * S == true_meters.

    Apply S to environment / prop transforms so the twin is 1:1 with reality.
    """
    t = float(true_meters)
    r = float(recon_length)
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"true_meters must be a positive finite number (got {true_meters})")
    if not math.isfinite(r) or r <= 1e-9:
        raise ValueError(f"recon_length must be a positive finite number (got {recon_length})")
    return t / r


def metric_scale_from_points(
    point_a: Sequence[float],
    point_b: Sequence[float],
    true_meters: float,
) -> Tuple[float, float]:
    """
    Scale from two reconstructed 3D points that correspond to a known real length.

    Returns (scale_factor, recon_length).
    """
    recon = euclidean_distance(point_a, point_b)
    return metric_scale_factor(true_meters=true_meters, recon_length=recon), recon


def resolve_metric_calibration(calibration: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalize API ``metric_calibration`` into a resolved dict with ``scale``.

    Supported modes:
      - reference_length: true_meters + recon_length
      - two_points: true_meters + point_a + point_b
      - player_height: true_meters (default 1.6) + recon_height

    Optional ``axis``:
      - ``uniform`` (default): scale X=Y=Z
      - ``horizontal``: scale X=Z only (keep Y / height) — for door-width fixes
    """
    if not calibration or not isinstance(calibration, dict):
        return None

    mode = str(calibration.get("mode") or "reference_length").strip().lower()
    axis = str(calibration.get("axis") or "uniform").strip().lower()
    if axis in ("xz", "horizontal_only", "floor_plan"):
        axis = "horizontal"
    if axis not in ("uniform", "horizontal"):
        axis = "uniform"

    true_meters = calibration.get("true_meters")
    if true_meters is None and mode == "player_height":
        true_meters = calibration.get("player_height_meters", 1.6)

    if mode in ("two_points", "points", "segment"):
        point_a = calibration.get("point_a")
        point_b = calibration.get("point_b")
        if not isinstance(point_a, (list, tuple)) or not isinstance(point_b, (list, tuple)):
            raise ValueError("two_points mode requires point_a and point_b arrays")
        scale, recon = metric_scale_from_points(point_a, point_b, float(true_meters))
        return {
            "mode": "two_points",
            "axis": axis,
            "true_meters": float(true_meters),
            "recon_length": recon,
            "scale": scale,
            "point_a": [float(x) for x in point_a[:3]],
            "point_b": [float(x) for x in point_b[:3]],
            "units": "meters",
            "one_to_one": True,
        }

    if mode in ("player_height", "height"):
        recon_height = calibration.get("recon_height")
        if recon_height is None:
            recon_height = calibration.get("recon_length")
        if recon_height is None:
            raise ValueError("player_height mode requires recon_height (measured height in recon units)")
        scale = metric_scale_factor(true_meters=float(true_meters), recon_length=float(recon_height))
        return {
            "mode": "player_height",
            "axis": axis,
            "true_meters": float(true_meters),
            "recon_length": float(recon_height),
            "scale": scale,
            "units": "meters",
            "one_to_one": True,
        }

    # default: reference_length
    recon_length = calibration.get("recon_length")
    if recon_length is None:
        raise ValueError(
            "reference_length mode requires recon_length (distance in reconstruction units) "
            "and true_meters (same distance in real meters, e.g. measured door width)"
        )
    if true_meters is None:
        raise ValueError("reference_length mode requires true_meters")
    scale = metric_scale_factor(true_meters=float(true_meters), recon_length=float(recon_length))
    return {
        "mode": "reference_length",
        "axis": axis,
        "true_meters": float(true_meters),
        "recon_length": float(recon_length),
        "scale": scale,
        "units": "meters",
        "one_to_one": True,
    }


def _as_scale_xyz(value: Any) -> Tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return float(value[0] or 1.0), float(value[1] or 1.0), float(value[2] or 1.0)
    if isinstance(value, dict):
        return (
            float(value.get("x", value.get("sx", 1.0)) or 1.0),
            float(value.get("y", value.get("sy", 1.0)) or 1.0),
            float(value.get("z", value.get("sz", 1.0)) or 1.0),
        )
    s = float(value or 1.0)
    return s, s, s


def apply_metric_scale_to_manifest(
    manifest: Dict[str, Any],
    calibration: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return a shallow-copied manifest with environment (+ props) scaled for 1:1 meters.

    With ``axis=horizontal``, only X/Z grow/shrink (height / Y unchanged).
    """
    resolved = resolve_metric_calibration(calibration) if calibration else None
    out = dict(manifest)
    meta = dict(out.get("metadata") or {})
    if not resolved:
        meta["metric_calibration"] = {"one_to_one": False, "reason": "not_provided"}
        out["metadata"] = meta
        return out

    scale = float(resolved["scale"])
    axis = str(resolved.get("axis") or "uniform")
    if axis == "horizontal":
        sx, sy, sz = scale, 1.0, scale
    else:
        sx = sy = sz = scale

    env = dict(out.get("environment") or {})
    transform = dict(env.get("transform") or {})
    px, py, pz = _as_scale_xyz(transform.get("scale", 1.0))
    transform["scale"] = [px * sx, py * sy, pz * sz]
    env["transform"] = transform
    out["environment"] = env

    props_out = []
    for prop in out.get("props") or []:
        if not isinstance(prop, dict):
            props_out.append(prop)
            continue
        p = dict(prop)
        t = dict(p.get("transform") or {})
        pos = t.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            t["position"] = [float(pos[0]) * sx, float(pos[1]) * sy, float(pos[2]) * sz]
        qx, qy, qz = _as_scale_xyz(t.get("scale", 1.0))
        t["scale"] = [qx * sx, qy * sy, qz * sz]
        p["transform"] = t
        props_out.append(p)
    if props_out:
        out["props"] = props_out

    spawn = dict(out.get("spawn") or {})
    # Keep player_height in real meters (do not scale the constant).
    if "player_height" not in spawn:
        spawn["player_height"] = 1.6
    out["spawn"] = spawn

    meta["metric_calibration"] = {
        **resolved,
        "scale_xyz": [sx, sy, sz],
    }
    meta["coordinate_units"] = "meters"
    out["metadata"] = meta
    out["coordinate_system"] = out.get("coordinate_system") or "y-up"
    return out


def summarize_calibration(resolved: Optional[Dict[str, Any]]) -> str:
    if not resolved:
        return "metric scale not applied"
    return (
        f"1:1 meters via {resolved.get('mode')}: "
        f"recon {resolved.get('recon_length'):.4g} → "
        f"{resolved.get('true_meters'):.4g} m (scale={resolved.get('scale'):.6g})"
    )
