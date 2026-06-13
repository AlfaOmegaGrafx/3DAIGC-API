"""
3D Mask Generation Utility

Creates 3D mask meshes (bounding boxes and ellipsoids) for use with
VoxHammer local mesh editing.
"""

import logging
import os
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class MaskGenerator:
    """
    Generator for 3D mask meshes used in local mesh editing.
    
    Supports creating procedural masks from geometric parameters:
    - Bounding box masks
    - Ellipsoid masks
    """
    
    @staticmethod
    def create_bbox_mask(
        center: List[float],
        dimensions: List[float],
        output_path: str,
        resolution: int = 20,
    ) -> str:
        """
        Create a bounding box mask mesh.
        
        Args:
            center: Center point [x, y, z]
            dimensions: Box dimensions [width, height, depth]
            output_path: Path to save the mask GLB file
            resolution: Mesh resolution (vertices per edge)
        
        Returns:
            Path to created mask file
        """
        if len(center) != 3:
            raise ValueError("center must be [x, y, z]")
        if len(dimensions) != 3:
            raise ValueError("dimensions must be [width, height, depth]")
        
        center = np.array(center, dtype=np.float32)
        dimensions = np.array(dimensions, dtype=np.float32)
        
        # Create box mesh
        logger.info(f"Creating bounding box mask: center={center}, dimensions={dimensions}")
        
        # Create a box using trimesh
        box = trimesh.creation.box(extents=dimensions)
        
        # Translate to center
        box.apply_translation(center)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export as GLB
        box.export(output_path)
        
        logger.info(f"Bounding box mask saved to: {output_path}")
        logger.info(f"Mask vertices: {len(box.vertices)}, faces: {len(box.faces)}")
        
        return output_path
    
    @staticmethod
    def create_ellipsoid_mask(
        center: List[float],
        radii: List[float],
        output_path: str,
        subdivisions: int = 3,
    ) -> str:
        """
        Create an ellipsoid mask mesh.
        
        Args:
            center: Center point [x, y, z]
            radii: Ellipsoid radii [rx, ry, rz]
            output_path: Path to save the mask GLB file
            subdivisions: Number of subdivisions for sphere (higher = smoother)
        
        Returns:
            Path to created mask file
        """
        if len(center) != 3:
            raise ValueError("center must be [x, y, z]")
        if len(radii) != 3:
            raise ValueError("radii must be [rx, ry, rz]")
        
        center = np.array(center, dtype=np.float32)
        radii = np.array(radii, dtype=np.float32)
        
        logger.info(f"Creating ellipsoid mask: center={center}, radii={radii}")
        
        # Create a unit sphere
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        
        # Scale to create ellipsoid
        # Create scaling matrix
        scale_matrix = np.diag(list(radii) + [1.0])
        sphere.apply_transform(scale_matrix)
        
        # Translate to center
        sphere.apply_translation(center)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export as GLB
        sphere.export(output_path)
        
        logger.info(f"Ellipsoid mask saved to: {output_path}")
        logger.info(f"Mask vertices: {len(sphere.vertices)}, faces: {len(sphere.faces)}")
        
        return output_path
    
    @staticmethod
    def _normalize_mask_params(
        mask_type: str, params: Union[List[float], dict[str, Any]]
    ) -> List[float]:
        if isinstance(params, dict):
            if mask_type.lower() in ["bbox", "box", "bounding_box"]:
                params = params.get("dimensions", params.get("size"))
            else:
                params = params.get("radii", params.get("radius"))
        if not isinstance(params, (list, tuple)) or len(params) != 3:
            key = "dimensions" if mask_type.lower() in ["bbox", "box", "bounding_box"] else "radii"
            raise ValueError(f"{key} must be [width, height, depth]" if key == "dimensions" else "radii must be [rx, ry, rz]")
        return [float(v) for v in params]

    @staticmethod
    def create_mask_from_params(
        mask_type: str,
        center: List[float],
        params: Union[List[float], dict[str, Any]],
        output_path: str,
        **kwargs,
    ) -> str:
        """
        Create a mask from parameters (convenience method).
        
        Args:
            mask_type: Type of mask ("bbox" or "ellipsoid")
            center: Center point [x, y, z]
            params: Dimensions/radii list, or dict with ``dimensions`` / ``radii`` keys
            output_path: Path to save the mask GLB file
            **kwargs: Additional parameters for mask creation
        
        Returns:
            Path to created mask file
        """
        params = MaskGenerator._normalize_mask_params(mask_type, params)
        if mask_type.lower() in ["bbox", "box", "bounding_box"]:
            return MaskGenerator.create_bbox_mask(
                center=center,
                dimensions=params,
                output_path=output_path,
                **kwargs,
            )
        elif mask_type.lower() in ["ellipsoid", "sphere", "ellipse"]:
            return MaskGenerator.create_ellipsoid_mask(
                center=center,
                radii=params,
                output_path=output_path,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown mask type: {mask_type}. "
                f"Supported types: bbox, ellipsoid"
            )
    
    @staticmethod
    def validate_mask_params(
        mask_type: str,
        center: List[float],
        params: List[float],
    ) -> Tuple[bool, str]:
        """
        Validate mask parameters.
        
        Args:
            mask_type: Type of mask
            center: Center point
            params: Mask parameters
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate center
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            return False, "center must be a list/tuple of 3 floats [x, y, z]"
        
        try:
            center = [float(x) for x in center]
        except (ValueError, TypeError):
            return False, "center values must be numeric"
        
        # Validate params
        if not isinstance(params, (list, tuple)) or len(params) != 3:
            return False, "params must be a list/tuple of 3 floats"
        
        try:
            params = [float(x) for x in params]
        except (ValueError, TypeError):
            return False, "params values must be numeric"
        
        # Check for positive values
        if any(x <= 0 for x in params):
            return False, "All dimension/radii values must be positive"
        
        # Validate mask type
        valid_types = ["bbox", "box", "bounding_box", "ellipsoid", "sphere", "ellipse"]
        if mask_type.lower() not in valid_types:
            return False, f"mask_type must be one of: {valid_types}"
        
        return True, ""
    
    @staticmethod
    def create_mask_visualization(
        mask_path: str,
        output_image_path: str,
        resolution: Tuple[int, int] = (800, 800),
    ) -> str:
        """
        Create a visualization of the mask mesh.
        
        Args:
            mask_path: Path to mask GLB file
            output_image_path: Path to save visualization image
            resolution: Image resolution (width, height)
        
        Returns:
            Path to visualization image
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # Load mask mesh
            mask = trimesh.load(mask_path)
            
            # Create figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh
            mesh_collection = Poly3DCollection(
                mask.vertices[mask.faces],
                alpha=0.3,
                facecolor='cyan',
                edgecolor='blue',
            )
            ax.add_collection3d(mesh_collection)
            
            # Set limits
            bounds = mask.bounds
            ax.set_xlim(bounds[0, 0], bounds[1, 0])
            ax.set_ylim(bounds[0, 1], bounds[1, 1])
            ax.set_zlim(bounds[0, 2], bounds[1, 2])
            
            # Labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Mask Visualization')
            
            # Save
            plt.savefig(output_image_path, dpi=resolution[0]/10)
            plt.close()
            
            logger.info(f"Mask visualization saved to: {output_image_path}")
            return output_image_path
            
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
            return ""
        except Exception as e:
            logger.error(f"Failed to create mask visualization: {str(e)}")
            return ""


def create_bbox_mask(
    center: List[float],
    dimensions: List[float],
    output_path: str,
    **kwargs,
) -> str:
    """Convenience function to create bounding box mask."""
    return MaskGenerator.create_bbox_mask(center, dimensions, output_path, **kwargs)


def create_ellipsoid_mask(
    center: List[float],
    radii: List[float],
    output_path: str,
    **kwargs,
) -> str:
    """Convenience function to create ellipsoid mask."""
    return MaskGenerator.create_ellipsoid_mask(center, radii, output_path, **kwargs)

