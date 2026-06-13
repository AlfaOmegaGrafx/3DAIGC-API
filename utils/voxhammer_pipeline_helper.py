"""
VoxHammer Pipeline Helper

Wrapper for VoxHammer local mesh editing pipeline that integrates with TRELLIS.
Supports text-guided and image-guided local 3D editing.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class VoxHammerInferenceHelper:
    """
    Helper class for VoxHammer local mesh editing pipeline.
    
    Wraps the complete 4-step VoxHammer pipeline:
    1. 3D Rendering: Generate multi-view images of the mesh
    2. Feature Extraction: Extract DINOv2 features from rendered images
    3. Voxel Masking: Generate voxel mask for editing region
    4. 3D Editing: Apply TRELLIS-based editing to the masked region
    """
    
    def __init__(
        self,
        voxhammer_root: str,
        trellis_pipeline=None,
        is_text_mode: bool = False,
    ):
        """
        Initialize VoxHammer helper.
        
        Args:
            voxhammer_root: Root directory of VoxHammer codebase
            trellis_pipeline: Pre-loaded TRELLIS pipeline (text or image mode)
            is_text_mode: Whether to use text-guided editing (vs image-guided)
        """
        self.voxhammer_root = Path(voxhammer_root)
        self.trellis_pipeline = trellis_pipeline
        self.is_text_mode = is_text_mode
        self.flux_pipeline = None
        
        # Import VoxHammer modules
        import sys
        if str(self.voxhammer_root) not in sys.path:
            sys.path.insert(0, str(self.voxhammer_root))
        
        # Add utils to path for rendering/inpainting
        utils_path = str(self.voxhammer_root / "utils")
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)
        
        logger.info(f"Initialized VoxHammer helper at: {voxhammer_root}")

    def _load_flux_pipeline(self):
        """Lazily load Flux inpainting pipeline."""
        if self.flux_pipeline is None:
            logger.info("Loading Flux Fill pipeline for guidance image inpainting...")
            import torch
            from diffusers import FluxFillPipeline
            self.flux_pipeline = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev", 
                torch_dtype=torch.bfloat16
            ).to("cuda")
        return self.flux_pipeline

    def generate_guidance_images(
        self,
        input_mesh_path: str,
        mask_glb_path: str,
        image_dir: str,
        target_prompt: str,
    ) -> Dict[str, str]:
        """
        Generate 2d_render.png, 2d_mask.png, and 2d_edit.png using VoxHammer utils.
        
        Args:
            input_mesh_path: Path to input 3D model
            mask_glb_path: Path to mask GLB
            image_dir: Directory to save generated images
            target_prompt: Prompt for inpainting
            
        Returns:
            Dictionary with paths to generated images
        """
        try:
            from thirdparty.VoxHammer.utils.render_rgb_and_mask import process as render_process
            from thirdparty.VoxHammer.utils.inpaint import inpaint as flux_inpaint
            
            os.makedirs(image_dir, exist_ok=True)
            
            # Step 1: Render multiple views
            logger.info(f"Rendering RGB and mask views to {image_dir}...")
            render_process(input_mesh_path, mask_glb_path, image_dir)
            
            # Step 2: Select a view for inpainting (usually 0000 is a good front view)
            # render_rgb_and_mask.py saves images as render_0000.png and mask_0000.png
            render_path = os.path.join(image_dir, "render_0000.png")
            mask_path = os.path.join(image_dir, "mask_0000.png")
            
            if not os.path.exists(render_path) or not os.path.exists(mask_path):
                # Fallback to any render/mask pair if 0000 is missing
                import glob
                renders = sorted(glob.glob(os.path.join(image_dir, "images", "render_*.png")))
                masks = sorted(glob.glob(os.path.join(image_dir, "images", "mask_*.png")))
                if not renders or not masks:
                    raise RuntimeError("Failed to generate rendered views for guidance")
                render_path = renders[0]
                mask_path = masks[0]
            
            # Step 3: Inpaint the selected view
            logger.info(f"Inpainting {render_path} with prompt: '{target_prompt}'")
            edit_path = os.path.join(image_dir, "2d_edit.png")
            
            pipe = self._load_flux_pipeline()
            flux_inpaint(
                pipeline=pipe,
                image_path=render_path,
                mask_path=mask_path,
                output_path=edit_path,
                prompt=target_prompt
            )
            
            # Finalize: Ensure we have the standard names expected by VoxHammer inference
            final_render_path = os.path.join(image_dir, "2d_render.png")
            final_mask_path = os.path.join(image_dir, "2d_mask.png")
            
            shutil.copy(render_path, final_render_path)
            shutil.copy(mask_path, final_mask_path)
            
            return {
                "2d_render": final_render_path,
                "2d_mask": final_mask_path,
                "2d_edit": edit_path
            }
            
        except ImportError as e:
            logger.error(f"Failed to import VoxHammer utility: {e}")
            raise RuntimeError(f"VoxHammer utilities not found or dependencies missing: {e}")
        except Exception as e:
            logger.error(f"Error generating guidance images: {e}")
            raise
    
    def run_complete_pipeline(
        self,
        input_model_path: str,
        mask_glb_path: str,
        output_path: str,
        source_prompt: str = "",
        target_prompt: str = "",
        image_dir: Optional[str] = None,
        render_params: Optional[Dict[str, Any]] = None,
        feature_params: Optional[Dict[str, Any]] = None,
        mask_params: Optional[Dict[str, Any]] = None,
        edit_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete VoxHammer editing pipeline.
        
        Args:
            input_model_path: Path to input 3D model file
            mask_glb_path: Path to mask GLB file (defines editing region)
            output_path: Path for final output GLB file
            source_prompt: Text prompt describing the source model (text mode)
            target_prompt: Text prompt describing the target model (text mode)
            image_dir: Directory containing source/target/mask images (image mode)
            render_params: Parameters for 3D rendering
            feature_params: Parameters for feature extraction
            mask_params: Parameters for voxel masking
            edit_params: Parameters for 3D editing
        
        Returns:
            Dictionary containing pipeline results
        """
        render_dir = None
        try:
            from utils.voxhammer_blender_runner import (
                run_3d_rendering,
                run_3d_editing,
                run_feature_extraction,
                run_voxel_masking,
            )
            
            # Create temporary render directory
            render_dir = tempfile.mkdtemp(prefix="voxhammer_render_")
            
            logger.info("=" * 60)
            logger.info("STARTING VOXHAMMER 3D EDITING PIPELINE")
            logger.info("=" * 60)
            
            results = {
                "input_model": input_model_path,
                "mask_glb": mask_glb_path,
                "render_dir": render_dir,
                "final_output": output_path,
            }
            
            # Step 1: 3D Rendering
            logger.info("Step 1: Rendering 3D model...")
            render_results = run_3d_rendering(
                input_model_path,
                render_dir,
                **(render_params or {})
            )
            results["rendering"] = render_results
            
            # Step 2: Feature Extraction
            logger.info("Step 2: Extracting features...")
            feature_results = run_feature_extraction(
                render_dir,
                **(feature_params or {})
            )
            results["features"] = feature_results
            
            # Step 3: Voxel Masking
            logger.info("Step 3: Generating voxel mask...")
            mask_results = run_voxel_masking(
                mask_glb_path,
                render_dir,
                **(mask_params or {})
            )
            results["masking"] = mask_results
            
            # Step 4: 3D Editing
            logger.info("Step 4: Performing 3D editing...")
            edit_results = run_3d_editing(
                self.trellis_pipeline,
                render_dir,
                output_path,
                image_dir or "assets/example/images",  # Default image dir
                self.is_text_mode,
                source_prompt,
                target_prompt,
                **(edit_params or {})
            )
            results["editing"] = edit_results
            
            logger.info("=" * 60)
            logger.info("VOXHAMMER PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Final result: {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"VoxHammer pipeline failed: {str(e)}")
            raise
        finally:
            # Cleanup render directory
            if render_dir and os.path.exists(render_dir):
                try:
                    shutil.rmtree(render_dir)
                    logger.info(f"Cleaned up render directory: {render_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup render directory: {e}")
    
    def edit_mesh_with_text(
        self,
        input_mesh_path: str,
        mask_glb_path: str,
        output_path: str,
        source_prompt: str,
        target_prompt: str,
        num_views: int = 150,
        resolution: int = 512,
        **kwargs,
    ) -> str:
        """
        Edit mesh using text guidance.
        
        Args:
            input_mesh_path: Path to input mesh
            mask_glb_path: Path to mask defining editing region
            output_path: Path for output mesh
            source_prompt: Text describing the original mesh
            target_prompt: Text describing the desired edited mesh
            num_views: Number of views for rendering
            resolution: Rendering resolution
            **kwargs: Additional parameters
        
        Returns:
            Path to edited mesh
        """
        render_params = {
            "num_views": num_views,
            "resolution": resolution,
            "engine": "CYCLES",
        }
        
        edit_params = {
            "skip_step": 0,
            "re_init": False,
            "cfg": [5.0, 6.0, 0.0, 0.0],  # CFG strengths
        }
        edit_params.update(kwargs)
        
        results = self.run_complete_pipeline(
            input_model_path=input_mesh_path,
            mask_glb_path=mask_glb_path,
            output_path=output_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            render_params=render_params,
            edit_params=edit_params,
        )
        
        return results["final_output"]
    
    def edit_mesh_with_image(
        self,
        input_mesh_path: str,
        mask_glb_path: str,
        output_path: str,
        image_dir: str,
        target_prompt: str = "",
        num_views: int = 150,
        resolution: int = 512,
        **kwargs,
    ) -> str:
        """
        Edit mesh using image guidance.
        
        Args:
            input_mesh_path: Path to input mesh
            mask_glb_path: Path to mask defining editing region
            output_path: Path for output mesh
            image_dir: Directory containing or to contain 2d_render.png, 2d_edit.png, 2d_mask.png
            target_prompt: Prompt for automated image generation (if images missing)
            num_views: Number of views for rendering
            resolution: Rendering resolution
            **kwargs: Additional parameters
        
        Returns:
            Path to edited mesh
        """
        # Check if required images exist, if not try to generate them
        required_images = ["2d_render.png", "2d_edit.png", "2d_mask.png"]
        missing_images = [
            img_name for img_name in required_images 
            if not os.path.exists(os.path.join(image_dir, img_name))
        ]
        
        if missing_images:
            if not target_prompt:
                raise ValueError(
                    f"Required images {missing_images} missing in {image_dir} "
                    "and no target_prompt provided for automated generation."
                )
            logger.info(f"Guidance images missing, generating them using target_prompt: '{target_prompt}'")
            self.generate_guidance_images(
                input_mesh_path=input_mesh_path,
                mask_glb_path=mask_glb_path,
                image_dir=image_dir,
                target_prompt=target_prompt
            )
        
        render_params = {
            "num_views": num_views,
            "resolution": resolution,
            "engine": "CYCLES",
        }
        
        edit_params = {
            "skip_step": 0,
            "re_init": False,
            "cfg": [5.0, 6.0, 0.0, 0.0],
        }
        edit_params.update(kwargs)
        
        results = self.run_complete_pipeline(
            input_model_path=input_mesh_path,
            mask_glb_path=mask_glb_path,
            output_path=output_path,
            source_prompt="",
            target_prompt="",
            image_dir=image_dir,
            render_params=render_params,
            edit_params=edit_params,
        )
        
        return results["final_output"]


def create_voxhammer_helper(
    trellis_pipeline,
    is_text_mode: bool = False,
    voxhammer_root: Optional[str] = None,
) -> VoxHammerInferenceHelper:
    """
    Convenience function to create VoxHammerInferenceHelper.
    
    Args:
        trellis_pipeline: Pre-loaded TRELLIS pipeline
        is_text_mode: Whether to use text-guided editing
        voxhammer_root: Root directory of VoxHammer (auto-detected if None)
    
    Returns:
        Initialized VoxHammerInferenceHelper
    """
    if voxhammer_root is None:
        voxhammer_root = os.path.join(os.getcwd(), "thirdparty", "VoxHammer")
    
    return VoxHammerInferenceHelper(
        voxhammer_root=voxhammer_root,
        trellis_pipeline=trellis_pipeline,
        is_text_mode=is_text_mode,
    )

