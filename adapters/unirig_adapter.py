"""
UniRig model adapter for automatic rigging of 3D meshes.

This adapter integrates the UniRig fast inference engine for automatic
mesh rigging with skeleton generation and skin weight computation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from core.models.base import ModelStatus
from core.models.rig_models import AutoRigModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.appearance_slots import (
    appearance_base_vrm_path,
    equip_slot_for_appearance,
    infer_appearance_slot,
    normalize_appearance_slot,
)
from core.utils.format_utils import (
    apply_appearance_component_rig,
    apply_humanoid_template_rig,
    apply_humanoid_template_wrap,
    extract_vrm_skeleton_fbx,
    fbx_to_glb,
    merge_rigged_fbx_with_source_mesh,
    source_mesh_has_textures,
)
from core.utils.humanoid_template import get_template, template_paths_available
from core.utils.mesh_utils import MeshProcessor

if TYPE_CHECKING:
    from utils.unirig_utils import UniRigInferenceEngine

logger = logging.getLogger(__name__)

# NOTE: Do NOT import utils.unirig_utils (or UniRig model graph) at module load.
# That pulls spconv → ninja JIT build (fails on this DGX CUDA / compute_52).
# template / template_wrap / appearance_component only need Blender; ML engine is lazy.


class UniRigAdapter(AutoRigModel):
    """
    Adapter for UniRig automatic rigging model.

    Integrates UniRig's fast inference engine for automatic mesh rigging
    with skeleton generation and skin weight computation.
    """

    def __init__(
        self,
        model_id: str = "unirig_auto_rig",
        model_path: Optional[str] = None,
        vram_requirement: int = 9216,  # 9GB VRAM
        unirig_root: Optional[str] = None,
        device: str = "cuda",
    ):
        if model_path is None:
            model_path = "pretrained/UniRig"

        if unirig_root is None:
            unirig_root = "thirdparty/UniRig"

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_input_formats=["fbx", "obj", "glb", "vrm"],
            supported_output_formats=["fbx", "glb", "vrm"],
        )

        self.unirig_root = Path(unirig_root)
        self.device = device
        self.inference_engine: Optional[UniRigInferenceEngine] = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

        # Verify UniRig installation
        if not self.unirig_root.exists():
            raise FileNotFoundError(f"UniRig not found at: {self.unirig_root}")

    def _load_model(self):
        """
        Mark adapter ready without loading UniRig weights.

        Template / appearance_component only need Blender. SkinTokens-style
        UniRig inference is loaded lazily on first skeleton/skin/full request
        (avoids spconv/ninja failures blocking clothing fits).
        """
        logger.info(
            "UniRig adapter ready (lazy inference engine; Blender paths available)"
        )
        return None

    def _ensure_inference_engine(self):
        """Load UniRig inference engine on demand (spconv/ninja-heavy — not for wrap)."""
        if self.inference_engine is not None:
            return self.inference_engine
        try:
            # Lazy import: avoids spconv JIT at UniRigAdapter construction / template_wrap.
            from utils.unirig_utils import InferenceConfig, UniRigInferenceEngine

            logger.info(f"Lazy-loading UniRig model from {self.unirig_root}")

            if str(self.unirig_root) not in sys.path:
                sys.path.insert(0, str(self.unirig_root))

            config = InferenceConfig(
                device=self.device,
                # torch.compile → ninja; broken on this DGX image (exit 2). Eager is fine.
                compile_model=False,
                cache_dir=str(
                    self.path_generator.base_output_dir / "temp" / "unirig_cache"
                ),
                precision="bf16-mixed",
            )
            self.inference_engine = UniRigInferenceEngine(config)
            self.inference_engine.preload_systems()
            logger.info("UniRig model loaded successfully")
            return self.inference_engine
        except Exception as e:
            logger.error(f"Failed to load UniRig model: {str(e)}")
            raise Exception(f"Failed to load UniRig model: {str(e)}") from e

    def _unload_model(self):
        """Unload UniRig model."""
        try:
            if self.inference_engine is not None:
                # Clear model cache
                self.inference_engine.clear_cache()
                self.inference_engine = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("UniRig model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading UniRig model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process auto-rigging request using UniRig.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - rig_mode: Rigging mode ("skeleton", "skin", "full") (default: "full")
                - output_format: Output format ("fbx", "glb", "vrm") (default: "fbx")
                - seed: Random seed for generation (default: None)
                - with_skinning: Whether to apply skinning weights (default: True)
                - skeleton_config: Path to skeleton task config (default: None)
                - skin_config: Path to skin task config (default: None)

        Returns:
            Dictionary with rigging results
        """
        try:
            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for auto-rigging")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            # Extract parameters early so appearance_component can skip UniRig engine.
            rig_mode = inputs.get("rig_mode", "full")
            if rig_mode not in ("template", "template_wrap", "appearance_component"):
                self._ensure_inference_engine()

            output_format = inputs.get("output_format", "fbx")
            seed = inputs.get("seed", None)
            with_skinning = inputs.get("with_skinning", True)
            skeleton_config = inputs.get("skeleton_config", None)
            skin_config = inputs.get("skin_config", None)
            humanoid_template_id = inputs.get("humanoid_template_id")
            appearance_slot = normalize_appearance_slot(
                inputs.get("appearance_slot")
            ) or infer_appearance_slot(
                object_name=inputs.get("object_name"),
                mesh_file_name=mesh_path.name,
            )

            # Validate rig mode
            allowed_modes = [
                "skeleton",
                "skin",
                "full",
                "template",
                "template_wrap",
                "appearance_component",
            ]
            if rig_mode not in allowed_modes:
                raise ValueError(
                    f"Invalid rig_mode: {rig_mode}. "
                    "Must be 'skeleton', 'skin', 'full', 'template', 'template_wrap', "
                    "or 'appearance_component' (clothing → Appearance Editor VRM fit)"
                )

            logger.info(f"Auto-rigging mesh with UniRig: {mesh_path}, mode: {rig_mode}")

            # Load and validate mesh
            try:
                mesh = self.mesh_processor.load_mesh(mesh_path)
                # if not self.mesh_processor.validate_mesh(mesh):
                # logger.warning("Input mesh validation failed, proceeding anyway")
                mesh_stats = self.mesh_processor.get_mesh_stats(mesh)
            except Exception as e:
                logger.warning(f"Failed to analyze input mesh: {e}")
                mesh_stats = {"vertex_count": 0, "face_count": 0}

            # Generate output paths
            base_name = mesh_path.stem
            output_path = self.path_generator.generate_rigged_path(
                self.model_id, base_name, output_format
            )
            output_dir = output_path.parent

            output_filename = output_path.name

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            wrap_requested = rig_mode == "template_wrap"
            appearance_requested = rig_mode == "appearance_component"
            rig_validation = None
            if appearance_requested:
                slot = appearance_slot or "Legs"
                base_vrm = appearance_base_vrm_path(
                    Path(__file__).resolve().parent.parent
                )
                if not base_vrm.is_file():
                    raise FileNotFoundError(
                        f"Appearance base VRM missing: {base_vrm}"
                    )
                glb_output = (
                    output_path if output_format == "glb" else output_path.with_suffix(".glb")
                )
                vrm_output = glb_output.with_suffix(".vrm")
                result_path, rig_validation = apply_appearance_component_rig(
                    str(base_vrm),
                    str(mesh_path),
                    str(glb_output),
                    appearance_slot=slot,
                    output_vrm_path=str(vrm_output),
                )
                # Prefer VRM as primary artifact so Appearance loadCustomTrait works.
                if Path(vrm_output).is_file():
                    result_path = str(vrm_output)
                has_skinning = True
                appearance_slot = slot
            elif rig_mode in ("template", "template_wrap"):
                template_id = humanoid_template_id or "template"
                spec = get_template(template_id)
                if not spec.vrm_path.is_file():
                    raise FileNotFoundError(f"Humanoid template VRM missing: {spec.vrm_path}")
                glb_output = (
                    output_path
                    if output_format == "glb"
                    else output_path.with_suffix(".glb")
                )
                vrm_output = (
                    output_path
                    if output_format == "vrm"
                    else glb_output.with_suffix(".vrm")
                )
                apply_fn = (
                    apply_humanoid_template_wrap
                    if wrap_requested
                    else apply_humanoid_template_rig
                )
                wrap_kwargs = {"output_vrm_path": str(vrm_output)}
                if wrap_requested:
                    mp = inputs  # model_parameters already merged into inputs
                    wrap_kwargs.update(
                        {
                            "gnm_identity": bool(mp.get("gnm_identity")),
                            "character_gender": mp.get("character_gender"),
                            "character_ethnicity": mp.get("character_ethnicity"),
                            "gnm_seed": mp.get("gnm_seed"),
                            "gnm_bake_expressions": bool(
                                mp.get("gnm_bake_expressions", mp.get("gnm_identity"))
                            ),
                            "gnm_replace_morphs": bool(mp.get("gnm_replace_morphs")),
                            "face_likeness": bool(mp.get("face_likeness")),
                            "likeness_alpha": float(mp.get("likeness_alpha", 0.65) or 0.65),
                            "likeness_source": mp.get("likeness_source") or "auto",
                            "likeness_image_path": mp.get("likeness_image_path")
                            or inputs.get("likeness_image_path"),
                        }
                    )
                    # Explicit Body+Cloth neck-open vs full-body mannequin (auto-detect if omitted).
                    if "expect_headless_body" in mp and mp.get("expect_headless_body") is not None:
                        wrap_kwargs["expect_headless_body"] = bool(mp.get("expect_headless_body"))
                    # Auto-enable GNM when ethnicity is provided from Studio.
                    if mp.get("character_ethnicity"):
                        wrap_kwargs["gnm_identity"] = True
                        wrap_kwargs["gnm_bake_expressions"] = bool(
                            mp.get("gnm_bake_expressions", True)
                        )
                result_path, rig_validation = apply_fn(
                    str(spec.vrm_path),
                    str(mesh_path),
                    str(glb_output),
                    **wrap_kwargs,
                )
                # Prefer VRM as primary download for Body+Cloth / template paths.
                if Path(vrm_output).is_file():
                    result_path = str(vrm_output)
                elif (rig_validation or {}).get("output_vrm") and Path(
                    str(rig_validation["output_vrm"])
                ).is_file():
                    result_path = str(rig_validation["output_vrm"])
                has_skinning = True
                # Downstream treats template_wrap like template for FBX skip.
                rig_mode = "template"
            elif rig_mode == "skeleton":
                # NOTE: using await here will make blender context incorrect
                result_path = self.inference_engine.generate_skeleton(
                    str(mesh_path),
                    str(output_dir),
                    os.path.join(
                        str(output_dir), output_filename
                    ),  # notice that this should ACTUALLY be output_path
                    skeleton_config,
                )
                has_skinning = False

            elif rig_mode == "skin":
                result_path = self.inference_engine.generate_skin_weights(
                    str(mesh_path),
                    str(output_dir),
                    os.path.join(str(output_dir), output_filename),
                    skin_config,
                )
                has_skinning = True

            else:  # full pipeline
                result_path = self.inference_engine.full_pipeline(
                    str(mesh_path),
                    str(output_dir),
                    os.path.join(str(output_dir), output_filename),
                )
                has_skinning = with_skinning

            # Convert rig FBX to GLB; preserve source textures when possible.
            if rig_mode not in ("template", "appearance_component"):
                rig_fbx_path = Path(result_path)
                if rig_fbx_path.suffix.lower() != ".fbx":
                    rig_fbx_path = rig_fbx_path.with_suffix(".fbx")
                glb_output = (
                    output_path
                    if output_format == "glb"
                    else rig_fbx_path.with_suffix(".glb")
                )
                if source_mesh_has_textures(mesh_path) and rig_fbx_path.is_file():
                    logger.info(
                        "Preserving source textures while attaching rig: %s",
                        mesh_path,
                    )
                    result_path = merge_rigged_fbx_with_source_mesh(
                        str(mesh_path),
                        str(rig_fbx_path),
                        str(glb_output),
                        apply_skinning=has_skinning,
                    )
                else:
                    result_path = fbx_to_glb(str(rig_fbx_path), str(glb_output))

            # Verify output was created
            if not Path(result_path).exists():
                raise Exception(f"UniRig failed to generate output file: {result_path}")

            # Estimate bone count from output (simplified approach)
            bone_count = self._estimate_bone_count(Path(result_path))

            if appearance_requested:
                generation_method = "appearance_component_vrm_fit"
                rig_type = "appearance_component"
            elif rig_mode == "template":
                generation_method = "humanoid_vrm_template"
                rig_type = "humanoid_template"
            else:
                generation_method = "unirig_fast_inference"
                rig_type = "auto_detected"
            rig_info = {
                "rig_type": rig_type,
                "has_skinning": has_skinning,
                "skeleton_only": rig_mode == "skeleton",
                "generation_method": generation_method,
                "bone_count": bone_count,
                "rig_mode": "appearance_component" if appearance_requested else rig_mode,
            }
            if appearance_requested:
                rig_info["appearance_slot"] = appearance_slot
                rig_info["equip_slot"] = equip_slot_for_appearance(appearance_slot or "Legs")
                rig_info["validation"] = rig_validation
                # GLB sibling kept next to VRM for debugging / non-VRM viewers
                glb_candidate = Path(str(result_path)).with_suffix(".glb")
                response_vrm = (
                    str(result_path) if str(result_path).lower().endswith(".vrm") else None
                )
                if not response_vrm and (rig_validation or {}).get("output_vrm"):
                    response_vrm = str(rig_validation["output_vrm"])
            elif rig_mode == "template":
                glb_candidate = Path(str(result_path)).with_suffix(".glb")
                if not glb_candidate.is_file() and (rig_validation or {}).get("output_glb"):
                    glb_candidate = Path(str(rig_validation["output_glb"]))
                response_vrm = (
                    str(result_path) if str(result_path).lower().endswith(".vrm") else None
                )
                if not response_vrm and (rig_validation or {}).get("output_vrm"):
                    response_vrm = str(rig_validation["output_vrm"])
            else:
                response_vrm = None
                glb_candidate = None
            if rig_mode == "template":
                rig_info["humanoid_template_id"] = humanoid_template_id or "template"
                rig_info["validation"] = rig_validation
                if wrap_requested:
                    rig_info["rig_mode"] = "template_wrap"
                    rig_info["wrap_status"] = (
                        (rig_validation or {}).get("wrap_status") or "head_stitch"
                    )
                    rig_info["wrap_humanoid_only"] = True
                    rig_info["blend_shapes_on_generated_mesh"] = bool(
                        (rig_validation or {}).get("blend_shapes_on_generated_mesh")
                    )
                    if (rig_validation or {}).get("morph_target_count") is not None:
                        rig_info["morph_target_count"] = rig_validation[
                            "morph_target_count"
                        ]

            response = {
                "output_mesh_path": str(result_path),
                "bone_count": bone_count,
                "rig_info": rig_info,
                "format": (
                    "vrm"
                    if appearance_requested or (response_vrm and rig_mode == "template")
                    else output_format
                ),
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "vertex_count": mesh_stats.get("vertex_count", 0),
                    "face_count": mesh_stats.get("face_count", 0),
                    "rig_mode": rig_info["rig_mode"],
                    "device": self.device,
                    "seed": seed,
                    "humanoid_template_id": humanoid_template_id,
                    "appearance_slot": appearance_slot,
                },
            }
            if response_vrm:
                response["output_vrm_path"] = response_vrm
                urls = {"vrm": response_vrm}
                if glb_candidate and Path(glb_candidate).is_file():
                    urls["glb"] = str(glb_candidate)
                    response["output_mesh_path_glb"] = str(glb_candidate)
                response["download_urls"] = urls
                # Keep output_mesh_path pointing at primary download (VRM).
                response["output_mesh_path"] = response_vrm
                response["format"] = "vrm"

            logger.info(f"UniRig auto-rigging completed: {result_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"UniRig auto-rigging failed: {str(e)}")
            raise Exception(f"UniRig auto-rigging failed: {str(e)}")

    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name

    def _estimate_bone_count(self, rigged_file: Path) -> int:
        """
        Estimate bone count from rigged file.

        This is a simplified implementation. In practice, you would
        parse the file format to count actual bones.
        """
        try:
            # For FBX files, we could parse and count bones
            # For now, return a reasonable estimate based on file size
            file_size = rigged_file.stat().st_size

            # Rough heuristic: larger files typically have more bones
            if file_size > 10 * 1024 * 1024:  # > 10MB
                return 50  # Complex rig
            elif file_size > 5 * 1024 * 1024:  # > 5MB
                return 30  # Medium rig
            elif file_size > 1 * 1024 * 1024:  # > 1MB
                return 20  # Simple rig
            else:
                return 15  # Minimal rig

        except Exception as e:
            logger.warning(f"Failed to estimate bone count: {e}")
            return 20  # Default estimate

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for UniRig."""
        return {"input": ["fbx", "obj", "glb"], "output": ["fbx", "glb"]}

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information for UniRig."""
        info = super().get_model_info()
        info.update(
            {
                "model_name": "UniRig",
                "version": "1.0",
                "description": "Unified automatic rigging using fast inference engine",
                "capabilities": [
                    "Automatic skeleton generation",
                    "Skin weight prediction",
                ],
                "stages": [
                    "Skeleton prediction using autoregressive transformer",
                    "Skin weight computation using bone-point cross attention",
                ],
                "requirements": {
                    "vram_gb": 8,
                    "pytorch_version": ">=2.3.1",
                    "cuda_required": True,
                },
                "interface": "fast_inference_engine",
                "supported_modes": ["skeleton", "skin", "full", "template"],
                "performance": {
                    "skeleton_generation": "~30-60 seconds",
                    "skin_generation": "~60-120 seconds",
                    "full_pipeline": "~90-180 seconds",
                },
            }
        )
        return info
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "rig_mode": {
                    "type": "string",
                    "description": (
                        "Rigging mode: skeleton, skin, full, template VRM fit, or "
                        "template_wrap (humanoid Phase 5 head stitch + optional GNM "
                        "identity / face likeness; not for creatures)"
                    ),
                    "default": "full",
                    "enum": ["skeleton", "skin", "full", "template", "template_wrap"],
                    "required": False,
                },
                "humanoid_template_id": {
                    "type": "string",
                    "description": "Template id when rig_mode is template or template_wrap (default: template → template.vrm)",
                    "default": "template",
                    "required": False,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                    "default": None,
                    "minimum": 0,
                    "required": False
                },
                "with_skinning": {
                    "type": "boolean",
                    "description": "Whether to apply skinning weights (only for full mode)",
                    "default": True,
                    "required": False
                },
                "gnm_identity": {
                    "type": "boolean",
                    "description": "Warp template head Basis toward GNM IdentitySampler (ethnicity/gender)",
                    "default": False,
                    "required": False,
                },
                "character_gender": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "GNM gender class for identity sampling",
                    "required": False,
                },
                "character_ethnicity": {
                    "type": "string",
                    "enum": ["asian", "black", "white", "middle_eastern"],
                    "description": "GNM ethnicity class for identity sampling",
                    "required": False,
                },
                "gnm_bake_expressions": {
                    "type": "boolean",
                    "description": "Bake additive GNM expression morphs onto template head",
                    "default": False,
                    "required": False,
                },
                "face_likeness": {
                    "type": "boolean",
                    "description": "Blend face likeness onto template (MeshMonk/RBF)",
                    "default": False,
                    "required": False,
                },
                "likeness_alpha": {
                    "type": "number",
                    "description": "Face likeness blend weight (0–1)",
                    "default": 0.65,
                    "minimum": 0,
                    "maximum": 1,
                    "required": False,
                },
                "likeness_source": {
                    "type": "string",
                    "enum": ["auto", "selfie", "body_roi"],
                    "description": (
                        "Likeness mesh source: selfie (MediaPipe from likeness_image), "
                        "body_roi (crop AIGC mesh), or auto (selfie if image provided)"
                    ),
                    "default": "auto",
                    "required": False,
                },
                "likeness_image_path": {
                    "type": "string",
                    "description": "Local path to selfie image for likeness_source=selfie/auto",
                    "required": False,
                },
                "likeness_image_file_id": {
                    "type": "string",
                    "description": "Uploaded selfie file id (resolved by API to likeness_image_path)",
                    "required": False,
                },
                "gnm_seed": {
                    "type": "integer",
                    "description": "RNG seed for GNM identity/expression sampling",
                    "required": False,
                },
            }
        }
