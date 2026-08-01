"""WorldMirror 2.0 (HY-World-2.0) helpers for multi-photo splat reconstruction."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
HYWORLD_ROOT = _REPO_ROOT / "thirdparty" / "HY-World-2.0"
DEFAULT_PRETRAINED = os.environ.get(
    "HYWORLD2_PRETRAINED", "tencent/HY-World-2.0"
)
DEFAULT_SUBFOLDER = os.environ.get(
    "HYWORLD2_MIRROR_SUBFOLDER", "HY-WorldMirror-2.0"
)


def ensure_hyworld_on_path() -> Path:
    """Add HY-World-2.0 to ``sys.path`` and return repo root."""
    root = HYWORLD_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HY-World-2.0 not found at {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def worldmirror2_repo_present() -> bool:
    pipeline_py = HYWORLD_ROOT / "hyworld2" / "worldrecon" / "pipeline.py"
    return pipeline_py.is_file()


def worldmirror2_importable() -> bool:
    if not worldmirror2_repo_present():
        return False
    try:
        ensure_hyworld_on_path()
        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # noqa: F401

        return True
    except Exception as exc:
        logger.debug("WorldMirror import check failed: %s", exc)
        return False


def worldmirror2_available() -> bool:
    return worldmirror2_importable()


def stage_images_directory(image_paths: Iterable[str]) -> Path:
    """Copy images into a temp directory (WorldMirror expects a folder or video)."""
    work = Path(tempfile.mkdtemp(prefix="worldmirror2_"))
    for index, raw in enumerate(image_paths):
        src = Path(raw)
        if not src.is_file():
            raise FileNotFoundError(f"Image not found: {src}")
        dest = work / f"view_{index:03d}{src.suffix.lower() or '.jpg'}"
        shutil.copy2(src, dest)
    return work


def find_gaussians_ply(output_dir: Path) -> Optional[Path]:
    direct = output_dir / "gaussians.ply"
    if direct.is_file():
        return direct
    for candidate in output_dir.rglob("gaussians.ply"):
        if candidate.is_file():
            return candidate
    return None


def ply_to_splat(ply_path: Path, splat_path: Path) -> Path:
    from plyfile import PlyData

    ensure_hyworld_on_path()
    from hyworld2.worldrecon.hyworldmirror.utils.save_utils import (
        process_ply_to_splat,
    )

    plydata = PlyData.read(str(ply_path))
    process_ply_to_splat(plydata, str(splat_path))
    return splat_path
