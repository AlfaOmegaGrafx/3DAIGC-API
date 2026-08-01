"""
Unit tests for trimesh quadric decimate adapter.
"""

from pathlib import Path

import pytest
import trimesh

from adapters.trimesh_decimate_adapter import TrimeshDecimateAdapter
from utils.mesh_decimate_utils import decimate_mesh, resolve_target_face_count


@pytest.fixture
def adapter():
    return TrimeshDecimateAdapter(
        model_id="trimesh_decimate",
        vram_requirement=256,
        default_target_face_count=500,
    )


@pytest.fixture
def sample_glb(tmp_path: Path) -> Path:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    path = tmp_path / "dense.glb"
    mesh.export(path)
    return path


class TestResolveTargetFaceCount:
    def test_explicit_face_count(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        assert resolve_target_face_count(mesh, target_face_count=100) == 100

    def test_ratio(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        n = len(mesh.faces)
        assert resolve_target_face_count(mesh, ratio=0.5) == max(1, int(round(n * 0.5)))

    def test_never_increases(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        n = len(mesh.faces)
        assert resolve_target_face_count(mesh, target_face_count=n * 10) == n


class TestDecimateMesh:
    def test_reduces_faces(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        before = len(mesh.faces)
        out, info = decimate_mesh(mesh, target_face_count=200)
        assert len(out.faces) <= 200
        assert len(out.faces) < before
        assert info["backend"] == "trimesh_quadric"
        assert info["skipped"] is False


class TestTrimeshDecimateAdapter:
    def test_process_request(self, adapter, sample_glb, tmp_path):
        out_glb = tmp_path / "decimated.glb"
        adapter.path_generator.generate_mesh_path = lambda *a, **k: out_glb
        adapter.path_generator.generate_info_path = lambda p: tmp_path / "info.json"

        result = adapter._process_request(
            {
                "mesh_path": str(sample_glb),
                "output_format": "glb",
                "target_face_count": 200,
            }
        )

        assert result["success"] is True
        assert Path(result["output_mesh_path"]).exists()
        assert result["retopology_info"]["operation"] == "decimate"
        assert result["output_stats"]["face_count"] <= 200
