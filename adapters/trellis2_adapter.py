"""
TRELLIS.2 model adapters for image-to-mesh generation and mesh painting.

TRELLIS.2 ONLY supports image-based operations (no text conditioning).
This adapter integrates TRELLIS.2 into our mesh generation framework.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.trellis2_utils import Trellis2Runner
from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel
from core.utils.thumbnail_utils import generate_mesh_thumbnail
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class Trellis2ImageToTexturedMeshAdapter(ImageToMeshModel):
    """
    Adapter for TRELLIS.2 image-to-mesh model.
    
    Integrates TRELLIS.2 for generating textured 3D meshes from images.
    Note: TRELLIS.2 does NOT support text conditioning.
    """
    
    FEATURE_TYPE = "image_to_textured_mesh"
    MODEL_ID = "trellis2_image_to_textured_mesh"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 12288,  # 12GB VRAM
        trellis2_root: Optional[str] = None,
    ):
        # Set default paths
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "pretrained", "TRELLIS.2")
            )
        
        if trellis2_root is None:
            trellis2_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "TRELLIS.2")
            )
        
        super().__init__(
            model_id=self.MODEL_ID,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=["glb", "obj"],
            feature_type=self.FEATURE_TYPE,
        )
        
        self.trellis2_root = Path(trellis2_root)
        self.model_path = Path(model_path)
        self.runner: Optional[Trellis2Runner] = None
        self.mesh_processor = MeshProcessor()
    
    def _load_model(self):
        """Load the TRELLIS.2 model pipeline."""
        try:
            logger.info(f"Loading TRELLIS.2 model from {self.trellis2_root}")
            
            # Initialize TRELLIS.2 runner
            self.runner = Trellis2Runner(
                trellis2_root=str(self.trellis2_root),
                model_cache_dir=str(self.model_path),
                device="cuda"
            )
            
            # Pre-load the image-to-3D pipeline
            self.runner._load_image_to_3d_pipeline()
            
            logger.info("TRELLIS.2 model loaded successfully")
            return self.runner
        
        except Exception as e:
            logger.error(f"Failed to load TRELLIS.2 model: {str(e)}")
            raise Exception(f"Failed to load TRELLIS.2 model: {str(e)}")
    
    def _unload_model(self):
        """Unload the TRELLIS.2 model."""
        try:
            if self.runner is not None:
                self.runner.cleanup()
                self.runner = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("TRELLIS.2 model unloaded successfully")
        
        except Exception as e:
            logger.error(f"Error unloading TRELLIS.2 model: {str(e)}")
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-mesh generation using TRELLIS.2.
        
        Args:
            inputs: Dictionary containing:
                - image_path: Path to input image (required)
                - decimation_target: Target number of faces (default: 1000000)
                - texture_size: Texture resolution (default: 4096)
                - remesh: Whether to remesh output (default: True)
                - remesh_band: Remesh band parameter (default: 1)
                - remesh_project: Remesh project parameter (default: 0)
                - seed: Random seed for reproducibility (default: None)
                - output_format: Output format (default: "glb")
        
        Returns:
            Dictionary with generated mesh information
        """
        try:
            if self.runner is None:
                raise ValueError("TRELLIS.2 model is not loaded")

            from core.utils.multi_image_input import (
                collect_local_image_paths,
                should_use_multiview_mesh,
            )

            image_path = inputs["image_path"]
            ref_paths = inputs.get("reference_image_paths") or []
            view_paths = (
                list(inputs["image_paths"])
                if inputs.get("image_paths")
                else collect_local_image_paths(image_path, ref_paths)
            )
            if should_use_multiview_mesh(view_paths, inputs) and len(view_paths) >= 2:
                logger.info(
                    "TRELLIS.2 job has %s views — delegating mesh to TRELLIS multiview",
                    len(view_paths),
                )
                from adapters.trellis_adapter import TrellisImageToTexturedMeshAdapter

                delegate = TrellisImageToTexturedMeshAdapter()
                delegate._load_model()
                return delegate._process_request(inputs)
            
            # Validate inputs using parent class
            output_format = self._validate_common_inputs(inputs)
            
            # Extract parameters
            decimation_target = inputs.get("decimation_target", 1000000)
            texture_size = inputs.get("texture_size", 4096)
            remesh = inputs.get("remesh", True)
            remesh_band = inputs.get("remesh_band", 1)
            remesh_project = inputs.get("remesh_project", 0)
            seed = inputs.get("seed", None)
            
            logger.info(f"Generating mesh with TRELLIS.2 for image: '{image_path}'")
            
            # Generate 3D mesh
            mesh = self.runner.image_to_mesh(
                image_path=image_path,
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=remesh,
                remesh_band=remesh_band,
                remesh_project=remesh_project,
                seed=seed,
                verbose=True
            )
            
            # Save mesh in requested format
            output_path = self._generate_output_path(image_path, output_format, is_prompt=False)
            self.mesh_processor.save_mesh(mesh, output_path, do_normalise=True)
            
            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail_path(output_path)
            thumbnail_generated = generate_mesh_thumbnail(
                str(output_path), str(thumbnail_path)
            )
            
            # Create response
            response = self._create_common_response(inputs, output_format)
            response.update({
                "output_mesh_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_generated else None,
                "generation_info": {
                    "model": "TRELLIS.2",
                    "image_path": image_path,
                    "seed": seed,
                    "vertex_count": len(mesh.vertices),
                    "face_count": len(mesh.faces),
                    "decimation_target": decimation_target,
                    "texture_size": texture_size,
                    "remesh": remesh,
                    "thumbnail_generated": thumbnail_generated,
                }
            })
            
            logger.info(f"TRELLIS.2 mesh generation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"TRELLIS.2 mesh generation failed: {str(e)}")
            raise Exception(f"TRELLIS.2 mesh generation failed: {str(e)}")
    
    def _generate_output_path(
        self, prompt: str, output_format: str, is_prompt: bool = True
    ) -> Path:
        """Generate output file path based on prompt and format."""
        # Create safe filename from prompt
        if is_prompt:
            safe_name = "".join(
                c for c in prompt[:50] if c.isalnum() or c in (" ", "_")
            ).strip()
            safe_name = safe_name.replace(" ", "_")
        else:
            safe_name = Path(prompt).stem[:50]  # Use filename stem for non-prompt inputs
        
        # Create output directory if it doesn't exist
        output_dir = Path(os.getcwd()) / "outputs" / "meshes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        filename = f"trellis2_{safe_name}_{timestamp}.{output_format}"
        
        return output_dir / filename
    
    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for TRELLIS.2."""
        return {"input": ["image"], "output": ["glb", "obj"]}
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "decimation_target": {
                    "type": "integer",
                    "description": "Target number of faces after decimation",
                    "default": 1000000,
                    "minimum": 10000,
                    "maximum": 10000000,
                    "required": False
                },
                "texture_size": {
                    "type": "integer",
                    "description": "Output texture resolution (width and height)",
                    "default": 4096,
                    "enum": [1024, 2048, 4096, 8192],
                    "required": False
                },
                "remesh": {
                    "type": "boolean",
                    "description": "Whether to remesh the output mesh",
                    "default": True,
                    "required": False
                },
                "remesh_band": {
                    "type": "integer",
                    "description": "Remesh band parameter (controls remeshing behavior)",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 10,
                    "required": False
                },
                "remesh_project": {
                    "type": "integer",
                    "description": "Remesh project parameter (controls projection)",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 10,
                    "required": False
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (None for random)",
                    "default": None,
                    "minimum": 0,
                    "required": False
                }
            }
        }


class Trellis2ImageMeshPaintingAdapter(ImageToMeshModel):
    """
    Adapter for TRELLIS.2 image-guided mesh texturing.
    
    Uses TRELLIS.2 to apply textures to meshes based on reference images.
    """
    
    FEATURE_TYPE = "image_mesh_painting"
    MODEL_ID = "trellis2_image_mesh_painting"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 12288,  # 12GB VRAM
        trellis2_root: Optional[str] = None,
    ):
        # Set default paths
        if model_path is None:
            model_path = os.path.join(os.getcwd(), "pretrained", "TRELLIS.2")
        
        if trellis2_root is None:
            trellis2_root = os.path.join(os.getcwd(), "thirdparty", "TRELLIS.2")
        
        super().__init__(
            model_id=self.MODEL_ID,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=["glb", "obj"],
            feature_type=self.FEATURE_TYPE,
        )
        
        self.trellis2_root = Path(trellis2_root)
        self.model_path = Path(model_path)
        self.runner: Optional[Trellis2Runner] = None
        self.mesh_processor = MeshProcessor()
    
    def _load_model(self):
        """Load the TRELLIS.2 texturing pipeline."""
        try:
            logger.info(f"Loading TRELLIS.2 texturing pipeline from {self.trellis2_root}")
            
            # Initialize TRELLIS.2 runner
            self.runner = Trellis2Runner(
                trellis2_root=str(self.trellis2_root),
                model_cache_dir=str(self.model_path),
                device="cuda"
            )
            
            # Pre-load the texturing pipeline
            self.runner._load_texturing_pipeline()
            
            logger.info("TRELLIS.2 texturing pipeline loaded successfully")
            return self.runner
        
        except Exception as e:
            logger.error(f"Failed to load TRELLIS.2 texturing pipeline: {str(e)}")
            raise Exception(f"Failed to load TRELLIS.2 texturing pipeline: {str(e)}")
    
    def _unload_model(self):
        """Unload the TRELLIS.2 model."""
        try:
            if self.runner is not None:
                self.runner.cleanup()
                self.runner = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("TRELLIS.2 texturing model unloaded successfully")
        
        except Exception as e:
            logger.error(f"Error unloading TRELLIS.2 texturing model: {str(e)}")
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-guided mesh texturing using TRELLIS.2.
        
        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - image_path: Path to reference image (required)
                - output_format: Output format (default: "glb")
                - extension_webp: Use WebP extension in GLB (default: True)
        
        Returns:
            Dictionary with textured mesh information
        """
        try:
            if self.runner is None:
                raise ValueError("TRELLIS.2 texturing model is not loaded")
            
            # Validate inputs
            output_format = self._validate_common_inputs(inputs)
            
            # Extract parameters
            mesh_path = inputs.get("mesh_path")
            image_path = inputs["image_path"]
            extension_webp = inputs.get("extension_webp", True)
            
            if not mesh_path:
                raise ValueError("mesh_path is required for mesh painting")
            
            logger.info(f"Texturing mesh with TRELLIS.2: mesh={mesh_path}, image={image_path}")
            
            # Apply texture to mesh
            textured_mesh = self.runner.texture_mesh(
                mesh_path=mesh_path,
                image_path=image_path,
                output_format=output_format,
                extension_webp=extension_webp
            )
            
            # Save mesh in requested format
            output_path = self._generate_output_path(mesh_path, output_format, is_prompt=False)
            self.mesh_processor.save_mesh(textured_mesh, output_path, do_normalise=True)
            
            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail_path(output_path)
            thumbnail_generated = generate_mesh_thumbnail(
                str(output_path), str(thumbnail_path)
            )
            
            # Create response
            response = self._create_common_response(inputs, output_format)
            response.update({
                "output_mesh_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_generated else None,
                "generation_info": {
                    "model": "TRELLIS.2-Texturing",
                    "mesh_path": mesh_path,
                    "image_path": image_path,
                    "vertex_count": len(textured_mesh.vertices),
                    "face_count": len(textured_mesh.faces),
                    "extension_webp": extension_webp,
                    "thumbnail_generated": thumbnail_generated,
                }
            })
            
            logger.info(f"TRELLIS.2 mesh texturing completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"TRELLIS.2 mesh texturing failed: {str(e)}")
            raise Exception(f"TRELLIS.2 mesh texturing failed: {str(e)}")
    
    def _generate_output_path(
        self, prompt: str, output_format: str, is_prompt: bool = True
    ) -> Path:
        """Generate output file path based on input and format."""
        if is_prompt:
            safe_name = "".join(
                c for c in prompt[:50] if c.isalnum() or c in (" ", "_")
            ).strip()
            safe_name = safe_name.replace(" ", "_")
        else:
            safe_name = Path(prompt).stem[:50]
        
        # Create output directory
        output_dir = Path(os.getcwd()) / "outputs" / "meshes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        filename = f"trellis2_textured_{safe_name}_{timestamp}.{output_format}"
        
        return output_dir / filename
    
    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for TRELLIS.2 texturing."""
        return {"input": ["mesh", "image"], "output": ["glb", "obj"]}
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "extension_webp": {
                    "type": "boolean",
                    "description": "Use WebP extension for textures in GLB format",
                    "default": True,
                    "required": False
                }
            }
        }

