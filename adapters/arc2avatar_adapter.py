"""
Arc2Avatar adapter (optional photoreal head track) — STUB.

Arc2Avatar generates FLAME-linked 3D Gaussian splat heads with blendshape-driven
expressions from a single image. It does NOT export standard mesh/VRM; output is
3DGS for rendering (e.g. Spark.js), not the template-rig body path.

Upstream: https://github.com/dimgerogiannis/Arc2Avatar
Paper: https://arc2avatar.github.io/

Enable when: thirdparty/Arc2Avatar is cloned, models downloaded, and license
review completed (FLAME + Arc2Face terms).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Arc2AvatarNotIntegratedError(NotImplementedError):
    """Raised until Arc2Avatar is installed and licensed for production."""


def get_arc2avatar_status() -> Dict[str, Any]:
    return {
        "integrated": False,
        "model_id": "arc2avatar_head_splat",
        "feature_type": "image_to_head_splat",
        "output_formats": ["ply", "spz"],
        "blendshapes": True,
        "flame_based": True,
        "vrm_export": False,
        "documentation": "docs/ARC2AVATAR_TRACK.md",
        "upstream": "https://github.com/dimgerogiannis/Arc2Avatar",
    }


def run_arc2avatar_inference(**_kwargs) -> Dict[str, Any]:
    raise Arc2AvatarNotIntegratedError(
        "Arc2Avatar is documented but not wired. See docs/ARC2AVATAR_TRACK.md"
    )
