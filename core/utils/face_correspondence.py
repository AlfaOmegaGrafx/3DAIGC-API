"""Face correspondence — MeshMonk pyramid wrap with RBF/NN fallback.

Transfers a source head surface onto a fixed template topology so morph
deltas remain valid:

  neutral' = F(neutral)
  morph'_i = neutral' + (morph_i - neutral)

Primary engine: ``meshmonk.pyramid_register`` (floating = template topo).
Fallback: landmark RBF / nearest-neighbor when MeshMonk is unavailable or
faces are missing.
"""
from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Prefer AvatarHead-like names when picking morph-bearing prims from template.vrm
HEAD_MESH_NAME_HINTS = (
    "avatarhead",
    "head",
    "face",
)
# Never treat accessory morph meshes as the primary wrap target / delta sink.
HEAD_MESH_NAME_EXCLUDE = (
    "eye",
    "tooth",
    "teeth",
    "hair",
    "lash",
    "brow",
    "tongue",
    "gum",
)


def _is_primary_head_mesh_name(name: str) -> bool:
    n = str(name or "").lower()
    if any(x in n for x in HEAD_MESH_NAME_EXCLUDE):
        return False
    return any(h in n for h in HEAD_MESH_NAME_HINTS)


def meshmonk_available() -> bool:
    try:
        import meshmonk  # noqa: F401

        return True
    except Exception:
        return False


def read_glb_chunks(path: str | Path) -> tuple[dict[str, Any], memoryview]:
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError(f"Invalid GLB: {path}")
    magic, _version, total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB/VRM: {path}")
    offset = 12
    gltf = None
    blob = None
    while offset + 8 <= min(total_len, len(data)):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:  # JSON
            gltf = json.loads(chunk)
        elif chunk_type == 0x004E4942:  # BIN
            blob = memoryview(chunk)
    if gltf is None or blob is None:
        raise ValueError(f"GLB missing JSON or BIN chunk: {path}")
    return gltf, blob


def _read_accessor(gltf: dict, blob: memoryview, accessor_idx: int) -> np.ndarray:
    acc = gltf["accessors"][accessor_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    off = (bv.get("byteOffset") or 0) + (acc.get("byteOffset") or 0)
    ctype = acc["componentType"]
    typ = acc["type"]
    count = int(acc["count"])
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[typ]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[ctype]
    arr = np.frombuffer(blob, dtype=dtype, count=count * n, offset=off)
    return arr.reshape(count, n).astype(np.float32)


def _faces_from_primitive(gltf: dict, blob: memoryview, prim: dict) -> Optional[np.ndarray]:
    """Decode triangle faces from a glTF primitive (TRIANGLES mode only)."""
    indices_idx = prim.get("indices")
    if indices_idx is None:
        return None
    mode = prim.get("mode", 4)  # 4 = TRIANGLES
    if mode not in (4, None):
        return None
    idx = _read_accessor(gltf, blob, int(indices_idx)).reshape(-1)
    if idx.size < 3 or idx.size % 3 != 0:
        return None
    return idx.reshape(-1, 3).astype(np.int32)


def load_template_head_neutral(
    template_vrm_path: str | Path,
) -> dict[str, Any]:
    """
    Load the primary morph-bearing head mesh POSITION (+ faces) from template.vrm.

    Returns dict with vertices, faces, mesh_index, prim_index, mesh_name, morph_count.
    """
    gltf, blob = read_glb_chunks(template_vrm_path)
    best = None
    for mi, mesh in enumerate(gltf.get("meshes") or []):
        name = str(mesh.get("name") or "")
        name_l = name.lower()
        for pi, prim in enumerate(mesh.get("primitives") or []):
            targets = prim.get("targets") or []
            pos_idx = (prim.get("attributes") or {}).get("POSITION")
            if pos_idx is None:
                continue
            verts = _read_accessor(gltf, blob, pos_idx)
            score = len(targets) * 10
            if _is_primary_head_mesh_name(name):
                score += 1000
                if "avatarhead" in name_l:
                    score += 500
            elif any(x in name_l for x in HEAD_MESH_NAME_EXCLUDE):
                score -= 500
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "vertices": verts,
                    "faces": _faces_from_primitive(gltf, blob, prim),
                    "mesh_index": mi,
                    "prim_index": pi,
                    "mesh_name": name,
                    "morph_count": len(targets),
                }
    if best is None or best["morph_count"] < 1:
        raise RuntimeError(f"No morph-bearing head mesh in {template_vrm_path}")
    best.pop("score", None)
    return best


def load_mesh_vertices_from_glb(
    path: str | Path,
    *,
    prefer_head: bool = True,
) -> np.ndarray:
    """Load a primary mesh POSITION array from GLB/VRM/GLTF container."""
    mesh = load_mesh_with_faces_from_glb(path, prefer_head=prefer_head)
    return mesh["vertices"]


def load_mesh_with_faces_from_glb(
    path: str | Path,
    *,
    prefer_head: bool = True,
) -> dict[str, Any]:
    """Load primary mesh vertices + triangle faces from GLB/VRM/GLTF."""
    gltf, blob = read_glb_chunks(path)
    best = None
    best_score = -1
    for mesh in gltf.get("meshes") or []:
        name_l = str(mesh.get("name") or "").lower()
        for prim in mesh.get("primitives") or []:
            pos_idx = (prim.get("attributes") or {}).get("POSITION")
            if pos_idx is None:
                continue
            verts = _read_accessor(gltf, blob, pos_idx)
            score = len(verts)
            if prefer_head and _is_primary_head_mesh_name(name_l):
                score += 100000
            elif prefer_head and any(x in name_l for x in HEAD_MESH_NAME_EXCLUDE):
                score -= 50000
            if score > best_score:
                best_score = score
                best = {
                    "vertices": verts,
                    "faces": _faces_from_primitive(gltf, blob, prim),
                    "mesh_name": str(mesh.get("name") or ""),
                }
    if best is None:
        raise RuntimeError(f"No mesh positions in {path}")
    return best


def _head_height_mask(vertices: np.ndarray, height_frac: float) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float32)
    spans = np.ptp(v, axis=0)
    axis = int(np.argmax(spans[1:])) + 1  # 1=Y or 2=Z
    lo, hi = float(v[:, axis].min()), float(v[:, axis].max())
    cut = hi - (hi - lo) * float(np.clip(height_frac, 0.15, 0.6))
    return v[:, axis] >= cut


def crop_head_region(
    vertices: np.ndarray,
    *,
    height_frac: float = 0.35,
) -> np.ndarray:
    """Keep the top ``height_frac`` of verts by Z (Y-up meshes also ok via max axis)."""
    v = np.asarray(vertices, dtype=np.float32)
    mask = _head_height_mask(v, height_frac)
    if int(mask.sum()) < 32:
        return v
    return v[mask]


def crop_head_mesh(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    *,
    height_frac: float = 0.35,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Crop head ROI and remap triangle indices; drops faces that leave the ROI."""
    v = np.asarray(vertices, dtype=np.float32)
    mask = _head_height_mask(v, height_frac)
    if int(mask.sum()) < 32:
        return v, None if faces is None else np.asarray(faces, dtype=np.int32)
    new_v = v[mask]
    if faces is None:
        return new_v, None
    f = np.asarray(faces, dtype=np.int32)
    keep = mask[f].all(axis=1)
    if int(keep.sum()) < 8:
        return new_v, None
    old_to_new = np.full(len(v), -1, dtype=np.int32)
    old_to_new[mask] = np.arange(int(mask.sum()), dtype=np.int32)
    return new_v, old_to_new[f[keep]]


def _face_landmarker_model_path() -> Path:
    env = (os.environ.get("MEDIAPIPE_FACE_LANDMARKER") or "").strip()
    if env and Path(env).is_file():
        return Path(env)
    # Repo default (downloaded by install / first use)
    return (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "mediapipe"
        / "face_landmarker.task"
    )


def selfie_image_to_face_mesh(
    image_path: str | Path,
    *,
    scale: float = 0.22,
) -> dict[str, np.ndarray]:
    """
    Build a triangular face mesh from a selfie via MediaPipe Face Landmarker.

    Returns dict with ``vertices`` (N,3) and ``faces`` (M,3) in a Y-up frame
    suitable as MeshMonk / RBF likeness source (caller should scale-align).
    """
    from PIL import Image
    from scipy.spatial import Delaunay

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Selfie image not found: {path}")

    model_path = _face_landmarker_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe Face Landmarker model missing: {model_path}. "
            "Download face_landmarker.task into assets/mediapipe/"
        )

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp

    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError(f"No face detected in selfie: {path}")

    lms = result.face_landmarks[0]
    verts = np.zeros((len(lms), 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        # MediaPipe: x/y normalized image coords, z roughly toward camera.
        verts[i, 0] = (float(lm.x) - 0.5) * float(scale)
        verts[i, 1] = -(float(lm.y) - 0.5) * float(scale)
        verts[i, 2] = -float(lm.z) * float(scale)

    # Triangulate on the frontal plane (x,y).
    tri = Delaunay(verts[:, :2])
    faces = np.asarray(tri.simplices, dtype=np.int32)
    return {"vertices": verts, "faces": faces, "source": "selfie_mediapipe"}


def rigid_scale_align(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Uniform-scale + translate source AABB to match target AABB."""
    src = np.asarray(source, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    src_min, src_max = src.min(axis=0), src.max(axis=0)
    tgt_min, tgt_max = tgt.min(axis=0), tgt.max(axis=0)
    src_c = (src_min + src_max) * 0.5
    tgt_c = (tgt_min + tgt_max) * 0.5
    src_size = np.maximum(src_max - src_min, 1e-6)
    tgt_size = np.maximum(tgt_max - tgt_min, 1e-6)
    scale = float(np.median(tgt_size / src_size))
    return (src - src_c) * scale + tgt_c


def head_bbox_diagonal(vertices: np.ndarray) -> float:
    v = np.asarray(vertices, dtype=np.float32)
    return float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))


def sanitize_warp_delta(
    delta: np.ndarray,
    template_neutral: np.ndarray,
    *,
    max_frac: float = 0.22,
    rms_frac: float = 0.07,
    min_keep_scale: float = 0.15,
    reject_if_clamped: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Clamp / reject extreme correspondence deltas that explode VRM head parts.

    Returns (safe_delta, info) where info includes ``clamped`` / ``rejected``.

    When ``reject_if_clamped`` is True, any out-of-bounds delta is zeroed
    (soft-clamped MeshMonk still shredded template.vrm eyes/teeth/hair).
    """
    d = np.asarray(delta, dtype=np.float32)
    info: dict[str, Any] = {
        "clamped": False,
        "rejected": False,
        "scale": 1.0,
        "max_norm": 0.0,
        "rms": 0.0,
        "diag": 0.0,
    }
    if d.size == 0:
        return d, info
    diag = max(head_bbox_diagonal(template_neutral), 1e-6)
    info["diag"] = diag
    norms = np.linalg.norm(d.reshape(-1, 3), axis=1)
    max_n = float(norms.max()) if len(norms) else 0.0
    rms = float(np.sqrt((d.astype(np.float64) ** 2).mean()))
    info["max_norm"] = max_n
    info["rms"] = rms
    max_allowed = diag * float(max_frac)
    rms_allowed = diag * float(rms_frac)
    if max_n <= max_allowed and rms <= rms_allowed:
        return d, info
    scale = 1.0
    if max_n > max_allowed:
        scale = min(scale, max_allowed / max(max_n, 1e-8))
    if rms > rms_allowed:
        scale = min(scale, rms_allowed / max(rms, 1e-8))
    info["scale"] = float(scale)
    if reject_if_clamped or scale < float(min_keep_scale):
        info["rejected"] = True
        info["clamped"] = True
        logger.warning(
            "Warp delta rejected (max=%.4f rms=%.4f diag=%.4f scale=%.3f "
            "reject_if_clamped=%s)",
            max_n,
            rms,
            diag,
            scale,
            reject_if_clamped,
        )
        return np.zeros_like(d), info
    info["clamped"] = True
    logger.warning(
        "Warp delta clamped (max=%.4f rms=%.4f diag=%.4f scale=%.3f)",
        max_n,
        rms,
        diag,
        scale,
    )
    return (d * scale).astype(np.float32), info


def _knn_indices(query: np.ndarray, cloud: np.ndarray, k: int = 3) -> np.ndarray:
    """Brute-force kNN indices (Nq, k). Fine for ~10k verts."""
    # (Nq, 1, 3) - (1, Nc, 3)
    # For large clouds subsample
    cloud_use = cloud
    if len(cloud) > 12000:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(cloud), size=12000, replace=False)
        cloud_use = cloud[pick]
    # chunk queries to limit memory
    k = max(1, min(k, len(cloud_use)))
    out = np.empty((len(query), k), dtype=np.int32)
    chunk = 512
    for i in range(0, len(query), chunk):
        q = query[i : i + chunk]
        d2 = ((q[:, None, :] - cloud_use[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        # sort the k
        row = np.arange(len(q))[:, None]
        order = np.argsort(d2[row, idx], axis=1)
        idx = idx[row, order]
        if len(cloud_use) != len(cloud):
            # map back through pick — approximate: use cloud_use indices only
            out[i : i + chunk] = idx
        else:
            out[i : i + chunk] = idx
    if len(cloud_use) != len(cloud):
        # Recompute against full cloud for final 1-NN only (quality)
        out1 = np.empty((len(query), 1), dtype=np.int32)
        for i in range(0, len(query), chunk):
            q = query[i : i + chunk]
            d2 = ((q[:, None, :] - cloud[None, :, :]) ** 2).sum(axis=2)
            out1[i : i + chunk, 0] = np.argmin(d2, axis=1)
        return out1
    return out


def nearest_neighbor_project(
    template_verts: np.ndarray,
    source_verts: np.ndarray,
) -> np.ndarray:
    """Project each template vertex onto nearest source vertex (after caller align)."""
    idx = _knn_indices(template_verts, source_verts, k=1)[:, 0]
    return source_verts[idx]


def rbf_warp(
    template_verts: np.ndarray,
    source_verts: np.ndarray,
    *,
    landmark_count: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """
    Thin landmark RBF: sample mutual landmarks, solve for TPS-like smooth warp.

    Falls back to nearest-neighbor if scipy is unavailable.
    """
    src = rigid_scale_align(source_verts, template_verts)
    tgt = np.asarray(template_verts, dtype=np.float32)
    n_land = min(landmark_count, len(tgt), len(src))
    rng = np.random.default_rng(seed)
    # Stratify by height for face coverage
    axis = int(np.argmax(np.ptp(tgt, axis=0)))
    order = np.argsort(tgt[:, axis])
    step = max(1, len(order) // n_land)
    land_t_idx = order[::step][:n_land]
    land_t = tgt[land_t_idx]
    land_s = nearest_neighbor_project(land_t, src)

    try:
        from scipy.interpolate import RBFInterpolator

        # Map template landmarks → source landmark positions
        rbf = RBFInterpolator(land_t, land_s, kernel="thin_plate_spline", smoothing=1e-3)
        warped = rbf(tgt).astype(np.float32)
        return warped
    except Exception as exc:
        logger.warning("RBF unavailable (%s); using nearest-neighbor project", exc)
        return nearest_neighbor_project(tgt, src)


def meshmonk_warp(
    template_verts: np.ndarray,
    source_verts: np.ndarray,
    *,
    template_faces: Optional[np.ndarray] = None,
    source_faces: Optional[np.ndarray] = None,
    num_iterations: int = 40,
    num_pyramid_layers: int = 3,
) -> np.ndarray:
    """
    Dense non-rigid wrap: deform template (floating) toward source (target).

    Returns warped template vertices (same count/order as ``template_verts``).
    Raises on missing MeshMonk / faces / registration failure.
    """
    import meshmonk
    import trimesh

    tv = np.asarray(template_verts, dtype=np.float32)
    sv = rigid_scale_align(source_verts, tv)
    tf = None if template_faces is None else np.asarray(template_faces, dtype=np.int32)
    sf = None if source_faces is None else np.asarray(source_faces, dtype=np.int32)
    if tf is None or len(tf) < 8:
        raise RuntimeError("meshmonk_warp requires template_faces")
    if sf is None or len(sf) < 8:
        raise RuntimeError("meshmonk_warp requires source_faces")

    floating = trimesh.Trimesh(vertices=tv, faces=tf, process=False)
    target = trimesh.Trimesh(vertices=sv, faces=sf, process=False)
    result = meshmonk.pyramid_register(
        floating=floating,
        target=target,
        rigid_params={"use_scaling": True, "num_iterations": 30},
        compute_normals_flag=True,
        num_iterations=int(num_iterations),
        num_pyramid_layers=int(num_pyramid_layers),
    )
    warped = np.asarray(result.aligned_vertices, dtype=np.float32)
    if warped.shape != tv.shape:
        raise RuntimeError(
            f"meshmonk output shape {warped.shape} != template {tv.shape}"
        )
    return warped


def deform_template_neutral(
    source_verts: np.ndarray,
    template_neutral: np.ndarray,
    *,
    method: str = "auto",
    alpha: float = 1.0,
    landmark_count: int = 128,
    template_faces: Optional[np.ndarray] = None,
    source_faces: Optional[np.ndarray] = None,
    engine_out: Optional[list] = None,
    reject_if_clamped: bool = False,
    max_frac: float = 0.22,
    rms_frac: float = 0.07,
) -> np.ndarray:
    """
    Return vertex delta (N,3) so ``template_neutral + delta`` matches source shape.

    ``alpha`` blends toward identity (0 = no change, 1 = full warp).

    Methods: ``auto`` (MeshMonk → RBF), ``meshmonk``, ``rbf``, ``nn``.
    If ``engine_out`` is a list, appends the resolved engine name.

    Cross-topology warps (GNM / MediaPipe → VRM) should use ``method="rbf"``
    and ``reject_if_clamped=True`` — dense MeshMonk shreds morph accessories.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 1e-6:
        if engine_out is not None:
            engine_out.append("none")
        return np.zeros_like(template_neutral, dtype=np.float32)
    method = (method or "auto").lower()
    used = method
    warped: Optional[np.ndarray] = None

    want_mm = method in ("auto", "meshmonk")
    if want_mm:
        if not meshmonk_available():
            if method == "meshmonk":
                raise RuntimeError("meshmonk package not installed")
            logger.info("MeshMonk unavailable; falling back to RBF")
            used = "rbf"
        elif template_faces is None or source_faces is None:
            if method == "meshmonk":
                raise RuntimeError("meshmonk requires template_faces and source_faces")
            logger.info("MeshMonk skipped (missing faces); falling back to RBF")
            used = "rbf"
        else:
            try:
                warped = meshmonk_warp(
                    template_neutral,
                    source_verts,
                    template_faces=template_faces,
                    source_faces=source_faces,
                )
                used = "meshmonk"
            except Exception as exc:
                if method == "meshmonk":
                    raise
                logger.warning("MeshMonk failed (%s); falling back to RBF", exc)
                used = "rbf"

    if warped is None:
        if used == "nn" or method == "nn":
            src = rigid_scale_align(source_verts, template_neutral)
            warped = nearest_neighbor_project(template_neutral, src)
            used = "nn"
        else:
            warped = rbf_warp(
                template_neutral, source_verts, landmark_count=landmark_count
            )
            used = "rbf"

    delta = (warped - template_neutral) * alpha
    delta, sanitize_info = sanitize_warp_delta(
        delta,
        template_neutral,
        max_frac=max_frac,
        rms_frac=rms_frac,
        reject_if_clamped=reject_if_clamped,
    )
    if sanitize_info.get("rejected"):
        used = f"{used}_rejected"
    elif sanitize_info.get("clamped"):
        used = f"{used}_clamped"
    if engine_out is not None:
        engine_out.append(used)
    logger.info(
        "deform_template_neutral method=%s alpha=%.2f delta_rms=%.5f clamped=%s",
        used,
        alpha,
        float(np.sqrt((delta**2).mean())),
        bool(sanitize_info.get("clamped")),
    )
    return delta.astype(np.float32)


def apply_morph_preserving_delta(
    neutral: np.ndarray,
    morph_abs_positions: list[np.ndarray] | tuple[np.ndarray, ...],
    delta: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Apply identity delta while preserving expression offsets:

      neutral' = neutral + delta
      morph'_i = neutral' + (morph_i - neutral)
    """
    neutral = np.asarray(neutral, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    new_neutral = neutral + delta
    new_morphs = []
    for morph in morph_abs_positions:
        m = np.asarray(morph, dtype=np.float32)
        new_morphs.append(new_neutral + (m - neutral))
    return new_neutral, new_morphs


def export_delta_npz(
    path: str | Path,
    delta: np.ndarray,
    *,
    mesh_name: str = "AvatarHead",
    mesh_index: int = -1,
    prim_index: int = 0,
    expression_deltas: Optional[dict[str, np.ndarray]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """Write NPZ consumed by Blender wrap script."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "delta": np.asarray(delta, dtype=np.float32),
        "mesh_name": np.asarray(mesh_name),
        "mesh_index": np.int32(mesh_index),
        "prim_index": np.int32(prim_index),
    }
    if expression_deltas:
        names = []
        for i, (name, arr) in enumerate(expression_deltas.items()):
            key = f"expr_{i}"
            payload[key] = np.asarray(arr, dtype=np.float32)
            names.append(name)
        payload["expr_names"] = np.asarray(names, dtype=object)
    if meta:
        payload["meta_json"] = np.asarray(json.dumps(meta), dtype=object)
    np.savez_compressed(out, **payload)
    return out


def transfer_gnm_deltas_to_template(
    gnm_deltas: dict[str, np.ndarray],
    gnm_neutral: np.ndarray,
    template_neutral: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Map per-expression GNM vertex deltas onto template topology.

    Builds correspondence once (template ← GNM neutral), then for each
    expression evaluates posed GNM and projects.
    """
    # Correspondence: template verts map to GNM indices via NN after align
    gnm_aligned = rigid_scale_align(gnm_neutral, template_neutral)
    idx = _knn_indices(template_neutral, gnm_aligned, k=1)[:, 0]
    out: dict[str, np.ndarray] = {}
    for name, delta_gnm in gnm_deltas.items():
        # delta on GNM → sample at mapped indices, then scale by alpha
        d = np.asarray(delta_gnm, dtype=np.float32)
        # When GNM was aligned by scale+translate only, deltas scale by same factor
        src_size = np.maximum(np.ptp(gnm_neutral, axis=0), 1e-6)
        tgt_size = np.maximum(np.ptp(template_neutral, axis=0), 1e-6)
        scale = float(np.median(tgt_size / src_size))
        out[name] = (d[idx] * scale * float(alpha)).astype(np.float32)
    return out
