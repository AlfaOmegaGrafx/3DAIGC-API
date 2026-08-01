"""
Unit tests for Instant Meshes retopology adapter.

Covers GLB/STL → OBJ conversion before the Instant Meshes CLI
(which only accepts .obj / .ply / .aln).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import trimesh

from adapters.instant_meshes_adapter import InstantMeshesRetopologyAdapter
from utils.instant_meshes_utils import INSTANT_MESHES_INPUT_EXTS


@pytest.fixture
def adapter():
    return InstantMeshesRetopologyAdapter(
        model_id="instant_meshes_retopology",
        vram_requirement=512,
        default_target_vertex_count=500,
    )


@pytest.fixture
def sample_glb(tmp_path: Path) -> Path:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    path = tmp_path / "creature.glb"
    mesh.export(path)
    return path


@pytest.fixture
def sample_obj(tmp_path: Path) -> Path:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    path = tmp_path / "creature.obj"
    mesh.export(path)
    return path


class TestInstantMeshesCliInputConversion:
    def test_obj_passes_through(self, adapter, sample_obj):
        mesh = adapter.mesh_processor.load_mesh(sample_obj)
        cli_input, tmp_dir = adapter._cli_input_for_mesh(sample_obj, mesh)
        assert cli_input == sample_obj
        assert tmp_dir is None

    def test_glb_converted_to_obj(self, adapter, sample_glb):
        mesh = adapter.mesh_processor.load_mesh(sample_glb)
        cli_input, tmp_dir = adapter._cli_input_for_mesh(sample_glb, mesh)
        try:
            assert cli_input.suffix.lower() == ".obj"
            assert cli_input.is_file()
            assert tmp_dir is not None
            assert cli_input.suffix.lower() in INSTANT_MESHES_INPUT_EXTS
            reloaded = trimesh.load(cli_input, force="mesh")
            assert len(reloaded.vertices) > 0
            assert len(reloaded.faces) > 0
        finally:
            if tmp_dir is not None:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_process_request_passes_obj_to_cli_for_glb(self, adapter, sample_glb, tmp_path):
        out_obj = tmp_path / "retopo.obj"

        def fake_run(mesh_path, output_path, **kwargs):
            mesh_path = Path(mesh_path)
            assert mesh_path.suffix.lower() in INSTANT_MESHES_INPUT_EXTS
            assert mesh_path.is_file()
            # Minimal valid OBJ for downstream load/save
            Path(output_path).write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
            return Path(output_path)

        with patch(
            "adapters.instant_meshes_adapter.run_instant_meshes", side_effect=fake_run
        ), patch.object(
            adapter.path_generator,
            "generate_mesh_path",
            return_value=out_obj,
        ), patch.object(
            adapter.path_generator,
            "generate_info_path",
            return_value=tmp_path / "info.json",
        ):
            result = adapter._process_request(
                {
                    "mesh_path": str(sample_glb),
                    "output_format": "obj",
                    "target_vertex_count": 200,
                }
            )

        assert result["success"] is True
        assert Path(result["output_mesh_path"]).exists()
