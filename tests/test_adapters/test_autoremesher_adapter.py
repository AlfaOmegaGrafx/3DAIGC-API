"""
Unit tests for AutoRemesher retopology adapter.

Covers GLB/STL → OBJ conversion before the AutoRemesher CLI
(which only accepts .obj).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import trimesh

from adapters.autoremesher_adapter import AutoRemesherRetopologyAdapter
from utils.autoremesher_utils import AUTOREMESHER_INPUT_EXTS


@pytest.fixture
def adapter():
    return AutoRemesherRetopologyAdapter(
        model_id="autoremesher_retopology",
        vram_requirement=512,
        default_target_quad_count=500,
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


class TestAutoRemesherCliInputConversion:
    def test_obj_is_preprocessed_to_temp(self, adapter, sample_obj):
        mesh = adapter.mesh_processor.load_mesh(sample_obj)
        cli_input, tmp_dir, meta = adapter._cli_input_for_mesh(sample_obj, mesh)
        try:
            assert cli_input != sample_obj
            assert cli_input.suffix.lower() == ".obj"
            assert cli_input.is_file()
            assert tmp_dir is not None
            assert meta["preprocess"] == "largest_component"
            assert meta["kept_component_faces"] > 0
        finally:
            if tmp_dir is not None:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_glb_converted_to_obj(self, adapter, sample_glb):
        mesh = adapter.mesh_processor.load_mesh(sample_glb)
        cli_input, tmp_dir, meta = adapter._cli_input_for_mesh(sample_glb, mesh)
        try:
            assert cli_input.suffix.lower() == ".obj"
            assert cli_input.is_file()
            assert tmp_dir is not None
            assert cli_input.suffix.lower() in AUTOREMESHER_INPUT_EXTS
            assert meta["preprocess"] == "largest_component"
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
            assert mesh_path.suffix.lower() in AUTOREMESHER_INPUT_EXTS
            assert mesh_path.is_file()
            Path(output_path).write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
            return Path(output_path)

        with patch(
            "adapters.autoremesher_adapter.run_autoremesher", side_effect=fake_run
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
                    "target_quad_count": 200,
                }
            )

        assert result["success"] is True
        assert Path(result["output_mesh_path"]).exists()
        assert result["retopology_info"]["backend"] == "autoremesher"

    def test_target_quad_count_overrides_vertex_count(self, adapter):
        assert adapter._resolve_target_quads(
            {"target_vertex_count": 1000, "target_quad_count": 2500}
        ) == 2500

    def test_vertex_count_maps_to_quads(self, adapter):
        assert adapter._resolve_target_quads({"target_vertex_count": 1200}) == 1200
