"""GNM Head helpers — identity / expression sampling and mesh export.

Uses vendored GNM Head (`thirdparty/GNM`, pull from
https://github.com/AlfaOmegaGrafx/GNM — fork of google/GNM) with NumPy mesh
eval + Keras semantic samplers (IdentitySampler / ExpressionSampler).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

GENDER_ALIASES = {
    "": None,
    "any": None,
    "male": "male",
    "m": "male",
    "man": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
}

ETHNICITY_ALIASES = {
    "": None,
    "any": None,
    "middle_eastern": "middle_eastern",
    "middle-eastern": "middle_eastern",
    "mideast": "middle_eastern",
    "asian": "asian",
    "white": "white",
    "caucasian": "white",
    "black": "black",
    "african": "black",
}

# GNM Expression enum → additive VRM / emotion morph names (Phase B).
GNM_EXPRESSION_TO_VRM = {
    "happy": "happy",
    "surprise": "surprised",
    "disgust": "angry",
    "squint": "gnm_squint",
    "smile_wide": "happy",
    "corners_down": "sad",
    "wink_left": "gnm_winkLeft",
    "wink_right": "gnm_winkRight",
    "pucker": "ou",
    "funneler": "oh",
    "blow": "ou",
    "snarl": "angry",
}

DEFAULT_BAKE_EXPRESSIONS = (
    "happy",
    "surprise",
    "disgust",
    "smile_wide",
    "corners_down",
    "pucker",
    "snarl",
)

_gnm_model = None
_identity_sampler = None
_expression_sampler = None


def normalize_gender(value: Any) -> Optional[str]:
    key = str(value or "").strip().lower()
    return GENDER_ALIASES.get(key, key if key in ("male", "female") else None)


def normalize_ethnicity(value: Any) -> Optional[str]:
    key = str(value or "").strip().lower().replace(" ", "_")
    if key in ETHNICITY_ALIASES:
        return ETHNICITY_ALIASES[key]
    if key in ("middle_eastern", "asian", "white", "black"):
        return key
    return None


def _gender_enum(gender: Optional[str]):
    from gnm.shape.semantic_sampler import Gender

    g = normalize_gender(gender) or "female"
    return Gender.MALE if g == "male" else Gender.FEMALE


def _ethnicity_enum(ethnicity: Optional[str]):
    from gnm.shape.semantic_sampler import Ethnicity

    e = normalize_ethnicity(ethnicity) or "white"
    return {
        "middle_eastern": Ethnicity.MIDDLE_EASTERN,
        "asian": Ethnicity.ASIAN,
        "white": Ethnicity.WHITE,
        "black": Ethnicity.BLACK,
    }[e]


def _expression_enum(label: str):
    from gnm.shape.semantic_sampler import Expression

    key = str(label or "").strip().upper().replace("-", "_").replace(" ", "_")
    if hasattr(Expression, key):
        return getattr(Expression, key)
    for member in Expression:
        if member.name.lower() == str(label or "").strip().lower():
            return member
    raise ValueError(f"Unknown GNM expression label: {label}")


def get_gnm_model():
    """Lazy-load GNM Head v3 (NumPy backend)."""
    global _gnm_model
    if _gnm_model is None:
        from gnm.shape import gnm_numpy

        _gnm_model = gnm_numpy.GNM.from_local(
            version=gnm_numpy.GNMMajorVersion.V3,
            variant=gnm_numpy.GNMVariant.HEAD,
        )
        logger.info(
            "GNM Head loaded identity_dim=%s expression_dim=%s verts=%s",
            _gnm_model.identity_dim,
            _gnm_model.expression_dim,
            len(_gnm_model.template_vertex_positions),
        )
    return _gnm_model


def get_identity_sampler():
    global _identity_sampler
    if _identity_sampler is None:
        from gnm.shape.semantic_sampler import IdentitySampler

        _identity_sampler = IdentitySampler()
    return _identity_sampler


def get_expression_sampler():
    global _expression_sampler
    if _expression_sampler is None:
        from gnm.shape.semantic_sampler import ExpressionSampler

        _expression_sampler = ExpressionSampler()
    return _expression_sampler


def sample_identity(
    gender: Optional[str] = None,
    ethnicity: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    num_samples: int = 1,
) -> np.ndarray:
    """Return identity coeff array shape (num_samples, identity_dim)."""
    rng = np.random.default_rng(seed)
    sampler = get_identity_sampler()
    return sampler.sample_identity(
        _gender_enum(gender),
        _ethnicity_enum(ethnicity),
        num_samples=num_samples,
        rng=rng,
    )


def sample_expression(
    label: str,
    *,
    intensity: float = 1.0,
    seed: Optional[int] = None,
    num_samples: int = 1,
) -> np.ndarray:
    """Return expression coeff array shape (num_samples, expression_dim)."""
    rng = np.random.default_rng(seed)
    sampler = get_expression_sampler()
    expr = sampler.sample_expression(
        _expression_enum(label),
        num_samples=num_samples,
        rng=rng,
    )
    intensity = float(np.clip(intensity, 0.0, 2.0))
    return expr * intensity


def evaluate_mesh(
    identity: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate GNM head mesh. Returns (vertices [V,3], triangles [F,3])."""
    gnm = get_gnm_model()
    id_vec = (
        np.zeros(gnm.identity_dim, dtype=np.float32)
        if identity is None
        else np.asarray(identity, dtype=np.float32).reshape(-1)[: gnm.identity_dim]
    )
    expr_vec = (
        np.zeros(gnm.expression_dim, dtype=np.float32)
        if expression is None
        else np.asarray(expression, dtype=np.float32).reshape(-1)[: gnm.expression_dim]
    )
    rotations = np.zeros((gnm.num_joints, 3), dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    verts = np.asarray(
        gnm(id_vec, expr_vec, rotations, translation),
        dtype=np.float32,
    )
    if verts.ndim == 3:
        verts = verts[0]
    faces = np.asarray(gnm.triangles, dtype=np.int32)
    return verts, faces


def export_mesh_npz(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    identity: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save vertices/faces (+ optional coeffs) for Blender / correspondence."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
    }
    if identity is not None:
        payload["identity"] = np.asarray(identity, dtype=np.float32)
    if expression is not None:
        payload["expression"] = np.asarray(expression, dtype=np.float32)
    if meta:
        import json

        payload["meta_json"] = np.asarray(json.dumps(dict(meta)), dtype=object)
    np.savez_compressed(out, **payload)
    return out


def generate_identity_mesh(
    gender: Optional[str] = None,
    ethnicity: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    output_npz: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Sample identity and evaluate neutral expression mesh."""
    identity = sample_identity(gender, ethnicity, seed=seed, num_samples=1)[0]
    verts, faces = evaluate_mesh(identity=identity, expression=None)
    result: dict[str, Any] = {
        "vertices": verts,
        "faces": faces,
        "identity": identity,
        "gender": normalize_gender(gender) or "female",
        "ethnicity": normalize_ethnicity(ethnicity) or "white",
        "seed": seed,
    }
    if output_npz:
        export_mesh_npz(
            output_npz,
            verts,
            faces,
            identity=identity,
            meta={
                "gender": result["gender"],
                "ethnicity": result["ethnicity"],
                "seed": seed,
            },
        )
        result["path"] = str(output_npz)
    return result


def generate_expression_mesh(
    identity: np.ndarray,
    label: str,
    *,
    intensity: float = 1.0,
    seed: Optional[int] = None,
    output_npz: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Evaluate GNM with fixed identity + sampled expression."""
    expression = sample_expression(label, intensity=intensity, seed=seed, num_samples=1)[0]
    verts, faces = evaluate_mesh(identity=identity, expression=expression)
    vrm_name = GNM_EXPRESSION_TO_VRM.get(str(label).strip().lower(), f"gnm_{label}")
    result: dict[str, Any] = {
        "vertices": verts,
        "faces": faces,
        "expression": expression,
        "label": label,
        "vrm_name": vrm_name,
        "intensity": intensity,
    }
    if output_npz:
        export_mesh_npz(
            output_npz,
            verts,
            faces,
            identity=identity,
            expression=expression,
            meta={"label": label, "vrm_name": vrm_name},
        )
        result["path"] = str(output_npz)
    return result


def bake_expression_deltas(
    identity: np.ndarray,
    labels: Optional[tuple[str, ...] | list[str]] = None,
    *,
    intensity: float = 1.0,
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Return {vrm_morph_name: delta_vertices} on GNM topology (posed - neutral)."""
    labels = tuple(labels or DEFAULT_BAKE_EXPRESSIONS)
    neutral, _faces = evaluate_mesh(identity=identity, expression=None)
    out: dict[str, np.ndarray] = {}
    for i, label in enumerate(labels):
        posed = generate_expression_mesh(
            identity,
            label,
            intensity=intensity,
            seed=(None if seed is None else int(seed) + i + 1),
        )["vertices"]
        name = GNM_EXPRESSION_TO_VRM.get(str(label).strip().lower(), f"gnm_{label}")
        if name in out:
            name = f"gnm_{label}"
        out[name] = (posed - neutral).astype(np.float32)
    return out
