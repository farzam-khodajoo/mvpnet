import copy
import logging
from typing import List
from pathlib import Path
from SETTING import SCANNET_DIRECTORY
from mvpnet.utils.visualize import SCANNET_COLOR_PALETTE

import open3d as o3d
import numpy as np
from pathlib import Path
from plyfile import PlyData

from .general import raise_path_error

logging.basicConfig(level=logging.INFO)

def unproject(k, depth_map, mask=None):
    "add intrinsic depth data into depth map"

    if mask is None:
        # only consider points where we have a depth value
        mask = depth_map > 0
    # create xy coordinates from image position
    v, u = np.indices(depth_map.shape)
    v = v[mask]
    u = u[mask]
    depth = depth_map[mask].ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    points_3d_xyz = (np.linalg.inv(k[:3, :3]).dot(uv1_points.T) * depth).T
    return points_3d_xyz


class PLYFormat:
    """
    Common IO to work with .ply files
    """

    @staticmethod
    def read_pc_from_ply(filename, return_color=False, return_label=False):
        """Read point clouds from ply files"""
        ply_data = PlyData.read(filename)
        vertex = ply_data["vertex"]
        x = np.asarray(vertex["x"])
        y = np.asarray(vertex["y"])
        z = np.asarray(vertex["z"])
        points = np.stack([x, y, z], axis=1)
        pc = {"points": points}
        if return_color:
            r = np.asarray(vertex["red"])
            g = np.asarray(vertex["green"])
            b = np.asarray(vertex["blue"])
            colors = np.stack([r, g, b], axis=1)
            pc["colors"] = colors
        if return_label:
            label = np.asarray(vertex["label"])
            pc["label"] = label
        return pc


class ScannetDataset:
    """
    Common utilites used to interact with Scannet dataset 
    (The path in SETTING.py must be correct)
    """

    @staticmethod
    def read_ids(dir=None):
        scannet_3d_dir = Path(SCANNET_DIRECTORY) if dir is None else Path(dir)
        if not scannet_3d_dir.exists():
            raise_path_error("3D scannet directory", scannet_3d_dir)

        dir_names = scannet_3d_dir.iterdir()
        dir_names: List[Path] = list(dir_names)  # convert to list from generator
        scan_ids = [name.name for name in dir_names]
        return scan_ids

    @staticmethod
    def get_scenes(dir=None):
        scene_dir = Path(SCANNET_DIRECTORY) if dir is None else Path(dir)
        scan_ids = ScannetDataset.read_ids(dir)
        return [scene_dir / scanid for scanid in scan_ids]
    

    @staticmethod
    def search_scanid(cache_data, scanid):
        id_query = [d for d in cache_data if d["scan_id"] == scanid]
        if len(id_query) == 0:
            raise_path_error("query scan id", scanid)

        return id_query[0]
    

class Common3D:
    """
    Common Open3D utilities
    """

    @staticmethod
    def create_bounding_box(pts):
        # Get the minimum and maximum coordinates of the point cloud
        min_bound, max_bound = pts.get_min_bound(), pts.get_max_bound()

        # Define the bounding box corners
        corners = np.array(
            [
                [min_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
            ]
        )

        # Create a line set to represent the bounding box
        lines = [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]

        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set line colors (optional)
        line_set.colors = o3d.utility.Vector3dVector(
            np.array([[0, 1, 0] for _ in range(len(lines))])
        )
        return line_set

    @staticmethod
    def set_color_for_overlaps_in_scene(scene_pts, scene_colors, overlap_pts, color):
        """Change a set of point in scene to specific color,
            this is used along with bounding box for segmented object
        """
        points_insize_bbox = np.all((overlap_pts.get_min_bound() <= scene_pts.points) & (scene_pts.points <= overlap_pts.get_max_bound()), axis=1)
        print(type(scene_colors))
        scene_colors = copy.deepcopy(scene_colors)

        if scene_colors.shape[1] == 3:
            scene_colors[points_insize_bbox] = color
            scene_pts.colors = o3d.utility.Vector3dVector(scene_colors)

        else:
            scene_colors[points_insize_bbox] = color
            scene_pts.colors = o3d.utility.Vector3dVector(scene_colors)

        return scene_pts
    
    @staticmethod
    def set_labels_for_overlaps_in_scene(scene_pts, scene_colors, overlap_pts, color_space):
        points_inside_bbox = np.all((overlap_pts.get_min_bound() <= scene_pts.points) & (scene_pts.points <= overlap_pts.get_max_bound()), axis=1)
        scene_colors[points_inside_bbox] = color_space[points_inside_bbox]
        scene_pts.colors = o3d.utility.Vector3dVector(scene_colors)
        return scene_pts

    @staticmethod
    def set_target_label(points: np.ndarray, colors: np.ndarray, seg_labels: np.ndarray, target_label: int, window_name="Open3D"):
        """
        Load colored scene while changing color of targeted label in the scene.
        visualize_target_label(points, seg_labels, target_label=1) # segment only the walls (label=1)
        """

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])

        indices = np.where(seg_labels == target_label)
        if target_label > len(SCANNET_COLOR_PALETTE):
            raise IndexError("Label {} does not exists.".format(target_label))
        rgb_color = np.array(SCANNET_COLOR_PALETTE[target_label]) / 255.
        colors[indices] = rgb_color
        pc.colors = o3d.utility.Vector3dVector(colors)

        return pc

    @staticmethod
    def visualize(*pts, window_name="Open3D"):
        o3d.visualization.draw_geometries([*pts], width=500, height=500, window_name=window_name)
