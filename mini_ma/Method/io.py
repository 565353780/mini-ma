import os
import cv2
import trimesh
import numpy as np
from typing import Optional


def loadImage(
    image_file_path: str,
    is_gray: bool=False,
) -> Optional[np.ndarray]:
    imread_flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR

    image_data = cv2.imread(image_file_path, imread_flag)

    if image_data is None:
        print('[ERROR][io::loadImage]')
        print('\t imread failed!')
        print('\t image_file_path:', image_file_path)
        return None

    return image_data

def getImageSize(image_file_path: str) -> list:
    image = cv2.imread(image_file_path)
    return image.shape[:2]

def loadMeshFile(
    mesh_file_path: str,
    color: list=[178, 178, 178],
) -> Optional[trimesh.Trimesh]:
    if not os.path.exists(mesh_file_path):
        print('[ERROR][io::loadMeshFile]')
        print('\t mesh file not exist!')
        print('\t mesh_file_path:', mesh_file_path)
        return None

    mesh = trimesh.load(mesh_file_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        print('[ERROR][NVDiffRastRenderer::loadMeshFile]')
        print('\t load mesh failed!')
        print('\t mesh_file_path:', mesh_file_path)
        return None

    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        num_verts = len(mesh.vertices)
        vertex_colors = np.tile(np.array(color), (num_verts, 1))
        mesh.visual.vertex_colors = vertex_colors

    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.compute_vertex_normals()

    return mesh
