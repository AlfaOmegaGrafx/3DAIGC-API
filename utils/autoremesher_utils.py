"""
AutoRemesher CLI wrapper (MIT).

Headless mode:
``autoremesher --input in.obj --output out.obj --target-quads 8000 ...``
See https://github.com/huxingyi/autoremesher
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import trimesh

logger = logging.getLogger(__name__)

AUTOREMESHER_INPUT_EXTS = frozenset({".obj"})

# AutoRemesher's halfedge builder aborts (SIGABRT / exit -6) on multi-component
# AIGC junk and non-manifold edges ("Found repeated halfedge" → double free).
_MIN_COMPONENT_FACES = 32

_REPO_REL = Path("thirdparty/autoremesher")
_BUILD_CANDIDATES = (
    _REPO_REL / "autoremesher",
    _REPO_REL / "build" / "autoremesher",
    _REPO_REL / "release" / "autoremesher",
)

_QT_STYLESHEET_RE = re.compile(
    r"QString::arg:.*?QPushButton:disabled\s*\{[^}]*\}",
    re.DOTALL,
)


def find_autoremesher_binary() -> Optional[Path]:
    """Resolve AutoRemesher executable (env, thirdparty build, or PATH)."""
    env_bin = os.environ.get("AUTOREMESHER_BIN")
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.is_file():
            return p.resolve()

    root = Path(__file__).resolve().parent.parent
    for candidate in _BUILD_CANDIDATES:
        p = (root / candidate).resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return p

    which = shutil.which("autoremesher")
    if which:
        return Path(which).resolve()
    return None


def prepare_mesh_for_autoremesher(
    mesh: trimesh.Trimesh,
    *,
    min_component_faces: int = _MIN_COMPONENT_FACES,
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Make an AIGC mesh safe for AutoRemesher's halfedge builder.

    Keeps the largest connected component (after dropping tiny debris),
    removes duplicate faces, and repairs normals. Multi-component / dusty
    GLBs otherwise crash the CLI with ``Found repeated halfedge`` / SIGABRT.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)!r}")

    work = mesh.copy()
    work.merge_vertices()
    work.update_faces(work.unique_faces())
    work.remove_unreferenced_vertices()

    components = work.split(only_watertight=False)
    if not components:
        raise ValueError("Mesh has no faces after cleanup")

    components.sort(key=lambda c: len(c.faces), reverse=True)
    kept = [c for c in components if len(c.faces) >= min_component_faces]
    if not kept:
        kept = [components[0]]

    # AutoRemesher is unstable on multi-component inputs — remesh the body only.
    primary = kept[0]
    primary = primary.copy()
    primary.update_faces(primary.unique_faces())
    primary.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(primary)
        trimesh.repair.fix_winding(primary)
    except Exception as exc:  # noqa: BLE001 — repair is best-effort
        logger.warning("AutoRemesher preprocess repair warning: %s", exc)

    if len(primary.faces) < 4:
        raise ValueError(
            "Mesh too small after AutoRemesher preprocess "
            f"({len(primary.faces)} faces)"
        )

    meta = {
        "input_vertices": int(len(mesh.vertices)),
        "input_faces": int(len(mesh.faces)),
        "component_count": len(components),
        "kept_component_faces": int(len(primary.faces)),
        "kept_component_vertices": int(len(primary.vertices)),
        "dropped_components": max(0, len(components) - 1),
        "preprocess": "largest_component",
    }
    logger.info(
        "AutoRemesher preprocess: %s faces → %s faces "
        "(%s components, dropped %s debris parts)",
        meta["input_faces"],
        meta["kept_component_faces"],
        meta["component_count"],
        meta["dropped_components"],
    )
    return primary, meta


def sanitize_autoremesher_error(text: str, *, max_len: int = 800) -> str:
    """Strip Qt stylesheet spam; keep halfedge / abort signals for the client."""
    if not text:
        return "no output"
    cleaned = _QT_STYLESHEET_RE.sub("", text)
    cleaned = re.sub(r"Q[A-Za-z]+::[^\n]{0,200}", "", cleaned)
    halfedges = re.findall(r"Found repeated halfedge:\S+", cleaned)
    markers = []
    for needle in (
        "double free or corruption",
        "Aborted",
        "Segmentation fault",
        "malloc",
    ):
        if needle.lower() in cleaned.lower():
            markers.append(needle)
    parts = []
    if markers:
        parts.append("; ".join(markers))
    if halfedges:
        sample = ", ".join(halfedges[:8])
        more = f" (+{len(halfedges) - 8} more)" if len(halfedges) > 8 else ""
        parts.append(f"non-manifold halfedges ({len(halfedges)}): {sample}{more}")
    if parts:
        summary = " | ".join(parts)
    else:
        summary = " ".join(cleaned.split())
    if len(summary) > max_len:
        summary = summary[: max_len - 3] + "..."
    return summary or "no output"


def run_autoremesher(
    mesh_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    target_quads: int = 8000,
    edge_scaling: float = 1.0,
    sharp_edge_degrees: float = 90.0,
    smooth_normal_degrees: float = 0.0,
    adaptivity: float = 1.0,
    report_path: Optional[Union[str, Path]] = None,
    timeout_sec: int = 3600,
) -> Path:
    """Run AutoRemesher headless and write ``output_path``."""
    binary = find_autoremesher_binary()
    if binary is None:
        raise FileNotFoundError(
            "AutoRemesher binary not found. Build with "
            "./scripts/install_autoremesher.sh or set AUTOREMESHER_BIN."
        )

    mesh_path = Path(mesh_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mesh_path.suffix.lower() not in AUTOREMESHER_INPUT_EXTS:
        raise ValueError(
            f"AutoRemesher requires OBJ input, got {mesh_path.suffix!r}"
        )

    with tempfile.TemporaryDirectory(prefix="ar_") as tmp:
        work_in = Path(tmp) / f"in{mesh_path.suffix}"
        work_out = Path(tmp) / f"out{output_path.suffix}"
        shutil.copy2(mesh_path, work_in)

        cmd = [
            str(binary),
            "--input",
            str(work_in),
            "--output",
            str(work_out),
            "--target-quads",
            str(int(target_quads)),
            "--edge-scaling",
            str(float(edge_scaling)),
            "--sharp-edge",
            str(float(sharp_edge_degrees)),
            "--smooth-normal",
            str(float(smooth_normal_degrees)),
            "--adaptivity",
            str(float(adaptivity)),
        ]
        if report_path is not None:
            work_report = Path(tmp) / "report.txt"
            cmd.extend(["--report", str(work_report)])

        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

        logger.info("Running AutoRemesher: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(binary.parent),
            env=env,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = sanitize_autoremesher_error(stderr or stdout)
            raise RuntimeError(
                f"AutoRemesher failed (exit {result.returncode}): {detail}"
            )

        if not work_out.is_file():
            raise RuntimeError(
                f"AutoRemesher did not produce output at {work_out}"
            )

        shutil.copy2(work_out, output_path)
        if report_path is not None and (Path(tmp) / "report.txt").is_file():
            shutil.copy2(Path(tmp) / "report.txt", Path(report_path))

    logger.info("AutoRemesher wrote %s", output_path)
    return output_path


def vertex_count_to_target_quads(target_vertex_count: int) -> int:
    """
    Map API vertex budget to AutoRemesher quad target.

    Quad meshes share vertices at edges; quad count is usually close to
    the desired vertex budget for character-scale assets.
    """
    return max(100, min(200_000, int(target_vertex_count)))


def get_autoremesher_info() -> Dict[str, Any]:
    binary = find_autoremesher_binary()
    return {
        "name": "autoremesher",
        "license": "MIT",
        "binary": str(binary) if binary else None,
        "available": binary is not None,
    }
