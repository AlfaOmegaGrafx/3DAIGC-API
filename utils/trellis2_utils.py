"""
TRELLIS.2 utility wrapper for mesh generation and texturing.

This module provides a high-level interface to TRELLIS.2 pipelines,
wrapping the inference logic from thirdparty/TRELLIS.2/example.py and example_texturing.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import torch
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


class Trellis2Runner:
    """
    Wrapper for TRELLIS.2 inference pipelines.
    
    Supports:
    - Image to 3D mesh generation
    - Image-guided mesh texturing
    """
    
    def __init__(
        self,
        trellis2_root: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize TRELLIS.2 runner.
        
        Args:
            trellis2_root: Path to thirdparty/TRELLIS.2 directory
            model_cache_dir: Path to model cache directory
            device: Device to run on ('cuda' or 'cpu')
        """
        if trellis2_root is None:
            trellis2_root = os.path.join(
                os.getcwd(), "thirdparty", "TRELLIS.2"
            )
        
        self.trellis2_root = Path(trellis2_root)
        self.device = device
        
        if model_cache_dir is None:
            model_cache_dir = os.path.join(
                os.getcwd(), "pretrained", "TRELLIS.2"
            )
        self.model_cache_dir = Path(model_cache_dir)

        # Prefer locally cached official microsoft/TRELLIS.2-4B weights; fall back to HF.
        _local_pipeline = self.model_cache_dir / "TRELLIS.2-4B"
        if (_local_pipeline / "pipeline.json").exists():
            self.pipeline_source = str(_local_pipeline)
        else:
            self.pipeline_source = "microsoft/TRELLIS.2-4B"
        
        # Add TRELLIS.2 to Python path
        if str(self.trellis2_root) not in sys.path:
            sys.path.insert(0, str(self.trellis2_root))
        
        # Set environment variables
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Pipelines (lazy loaded)
        self.image_to_3d_pipeline = None
        self.texturing_pipeline = None
        self.envmap = None
        
        # Import o_voxel for GLB export
        try:
            import o_voxel
            self.o_voxel = o_voxel
        except ImportError:
            logger.error("Failed to import o_voxel. GLB export will not be available.")
            self.o_voxel = None
    
    def _load_image_to_3d_pipeline(self):
        """Load TRELLIS.2 image-to-3D pipeline."""
        if self.image_to_3d_pipeline is not None:
            return
        
        try:
            logger.info("Loading TRELLIS.2 image-to-3D pipeline...")
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            
            self.image_to_3d_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                self.pipeline_source
            )
            
            if self.device == "cuda":
                self.image_to_3d_pipeline.cuda()
            
            logger.info("TRELLIS.2 image-to-3D pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TRELLIS.2 image-to-3D pipeline: {e}")
            raise
    
    def _load_texturing_pipeline(self):
        """Load TRELLIS.2 texturing pipeline."""
        if self.texturing_pipeline is not None:
            return
        
        try:
            logger.info("Loading TRELLIS.2 texturing pipeline...")
            from trellis2.pipelines import Trellis2TexturingPipeline
            
            self.texturing_pipeline = Trellis2TexturingPipeline.from_pretrained(
                self.pipeline_source,
                config_file="texturing_pipeline.json"
            )
            
            if self.device == "cuda":
                self.texturing_pipeline.cuda()
            
            logger.info("TRELLIS.2 texturing pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TRELLIS.2 texturing pipeline: {e}")
            raise
    
    def _load_envmap(self, envmap_path: Optional[str] = None):
        """Load environment map for rendering."""
        if self.envmap is not None:
            return
        
        try:
            from trellis2.renderers import EnvMap
            
            if envmap_path is None:
                # Use default environment map from TRELLIS.2 assets
                envmap_path = str(self.trellis2_root / "assets" / "hdri" / "forest.exr")
            
            if not os.path.exists(envmap_path):
                logger.warning(f"Environment map not found: {envmap_path}")
                return
            
            envmap_data = cv2.cvtColor(
                cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED),
                cv2.COLOR_BGR2RGB
            )
            
            self.envmap = EnvMap(
                torch.tensor(envmap_data, dtype=torch.float32, device=self.device)
            )
            
            logger.info(f"Environment map loaded from {envmap_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load environment map: {e}")
            self.envmap = None
    
    def image_to_mesh(
        self,
        image_path: str,
        decimation_target: int = 1000000,
        texture_size: int = 4096,
        remesh: bool = True,
        remesh_band: int = 1,
        remesh_project: int = 0,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> trimesh.Trimesh:
        """
        Generate 3D mesh from image using TRELLIS.2.
        
        Args:
            image_path: Path to input image
            decimation_target: Target number of faces after decimation
            texture_size: Output texture resolution
            remesh: Whether to remesh the output
            remesh_band: Remesh band parameter
            remesh_project: Remesh project parameter
            seed: Random seed for reproducibility
            verbose: Whether to show progress
        
        Returns:
            Generated trimesh.Trimesh object
        """
        self._load_image_to_3d_pipeline()
        
        # Load image
        image = Image.open(image_path)
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Run generation pipeline
        logger.info(f"Generating 3D mesh from image: {image_path}")
        mesh = self.image_to_3d_pipeline.run(image)[0]
        
        # Simplify mesh (nvdiffrast has a limit of 16777216 faces)
        mesh.simplify(16777216)
        
        # Export to GLB format
        if self.o_voxel is not None:
            logger.info("Exporting mesh to GLB format...")
            glb = self.o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=remesh,
                remesh_band=remesh_band,
                remesh_project=remesh_project,
                verbose=verbose
            )
            
            # Convert GLB to trimesh
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                glb.export(tmp.name)
                output_mesh = trimesh.load(tmp.name, force="mesh")
                os.unlink(tmp.name)
            
            return output_mesh
        else:
            logger.warning("o_voxel not available, returning raw mesh")
            # Return a basic conversion (without proper texture baking)
            return trimesh.Trimesh(
                vertices=mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices,
                faces=mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
            )
    
    def texture_mesh(
        self,
        mesh_path: str,
        image_path: str,
        output_format: str = "glb",
        extension_webp: bool = True
    ) -> trimesh.Trimesh:
        """
        Apply texture to mesh using image guidance.
        
        Args:
            mesh_path: Path to input mesh file
            image_path: Path to reference image for texturing
            output_format: Output format ('glb' or 'obj')
            extension_webp: Whether to use WebP extension in GLB
        
        Returns:
            Textured trimesh.Trimesh object
        """
        self._load_texturing_pipeline()
        
        # Load mesh and image
        mesh = trimesh.load(mesh_path, force="mesh")
        image = Image.open(image_path)
        
        # Run texturing pipeline
        logger.info(f"Texturing mesh with image: {image_path}")
        output = self.texturing_pipeline.run(mesh, image)
        
        # Export to temporary file and reload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
            output.export(tmp.name, extension_webp=extension_webp)
            textured_mesh = trimesh.load(tmp.name, force="mesh")
            os.unlink(tmp.name)
        
        return textured_mesh
    
    def cleanup(self):
        """Clean up resources and free GPU memory."""
        if self.image_to_3d_pipeline is not None:
            if hasattr(self.image_to_3d_pipeline, 'cpu'):
                self.image_to_3d_pipeline.cpu()
            del self.image_to_3d_pipeline
            self.image_to_3d_pipeline = None
        
        if self.texturing_pipeline is not None:
            if hasattr(self.texturing_pipeline, 'cpu'):
                self.texturing_pipeline.cpu()
            del self.texturing_pipeline
            self.texturing_pipeline = None
        
        if self.envmap is not None:
            del self.envmap
            self.envmap = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("TRELLIS.2 runner cleaned up")

