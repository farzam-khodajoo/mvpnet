from typing import List
import argparse
from pathlib import Path
from glob import glob
from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY
from mvpnet.utils.o3d_util import draw_point_cloud

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm


DEPTH_SCALE_FACTOR = 50.0


def raise_path_error(title, path=""):
    print("[Error] {} {} does not exists.".format(title, path))
    exit()


def read_ids():
    scannet_3d_dir = Path(SCANNET_DIRECTORY)
    if not scannet_3d_dir.exists():
        raise_path_error("3D scannet directory", scannet_3d_dir)

    dir_names = scannet_3d_dir.iterdir()
    dir_names: List[Path] = list(dir_names)  # convert to list from generator
    scan_ids = [name.name for name in dir_names]
    return scan_ids


def get_scenes():
    scene_dir = Path(SCANNET_DIRECTORY)
    scan_ids = read_ids()
    return [scene_dir / scanid for scanid in scan_ids]


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


def unproject(k, depth_map, mask=None):
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

def create_bounding_box(pts):
    # Get the minimum and maximum coordinates of the point cloud
    min_bound, max_bound = pts.get_min_bound(), pts.get_max_bound()


    # Define the bounding box corners
    corners = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]]
    ])

    # Create a line set to represent the bounding box
    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set line colors (optional)
    line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(len(lines))]))
    return line_set

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="sample which should be searched on dataset",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="visualize saved report",
    )
    args = parser.parse_args()

    target_dir = Path(args.target)
    save_dir = Path(args.target) / "report"

    if not target_dir.exists():
        raise_path_error("target folder", target_dir)

    # create report directory
    save_dir.mkdir(exist_ok=True)

    if args.report:
        scene_ply = glob(str(save_dir / "*.ply"))
        bbox = glob(str(save_dir / "*.npy"))
        
        if len(scene_ply) < 1:
            raise_path_error("report folder's .ply")

        if len(bbox) < 1:
            raise_path_error("report folder's .npy")

        pc = read_pc_from_ply(scene_ply[0])
        pts_vis = draw_point_cloud(pc["points"])

        bbox = np.load(bbox[0])
        overlap_point_cloud = draw_point_cloud(np.asarray(pts_vis.points)[bbox[:]])
        bbox = create_bounding_box(overlap_point_cloud)
        print("[Info] Loading scene")
        o3d.visualization.draw_geometries([bbox, pts_vis], height=500, width=500)
        exit()


    # check color sample
    sample_color_glob = glob(str(target_dir / "*.jpg"))
    if len(sample_color_glob) < 1:
        raise_path_error("color.jpg file")

    sample_color = sample_color_glob[0]

    # check depth sample
    sample_depth_glob = glob(str(target_dir / "*.png"))
    if len(sample_depth_glob) < 1:
        raise_path_error("color.jpg file")

    sample_depth = sample_depth_glob[0]

    color_sample_id = Path(sample_color).name.split(".jpg")[0]
    depth_sample_id = Path(sample_depth).name.split(".png")[0]

    if color_sample_id != depth_sample_id:
        print("color.jpg id and depth.jpg id mismatch !")
        print("color: {} --- depth: {}".format(color_sample_id, depth_sample_id))
        exit()

    sample_pose = target_dir / "{}.txt".format(depth_sample_id)
    if not sample_pose.exists():
        raise_path_error("pose txt file", sample_pose)

    sample_intrinsic_depth = target_dir / "intrinsic_depth.txt"
    if not sample_intrinsic_depth.exists():
        raise_path_error("intrinsic file", sample_intrinsic_depth)

    knn_distances = {}

    print("[Info] Searching scenes")
    for scene in tqdm(get_scenes()):
        # check if folder exists
        if not scene.exists():
            raise_path_error("scene", scene)

        sample_id = scene.name
        sample_ply_scene_path = scene / "{}_vh_clean_2.ply".format(sample_id)
        pc = read_pc_from_ply(sample_ply_scene_path, return_color=True)
        pts_vis = draw_point_cloud(pc["points"])

        # load image
        color = Image.open(sample_color)
        color = np.array(color)

        # loading instrinsic data
        cam_matrix = np.loadtxt(sample_intrinsic_depth, dtype=np.float32)

        # load depth image
        depth = Image.open(sample_depth)
        depth = np.asarray(depth, dtype=np.float32) / DEPTH_SCALE_FACTOR

        # load pose data
        pose = np.loadtxt(sample_pose)

        # unporoject 2D to 3D
        unproj_pts = unproject(cam_matrix, depth)
        unproj_pts = pose[:3, :3].dot(unproj_pts[:, :3].T).T + pose[:3, 3]
        unproj_pts_vis = draw_point_cloud(unproj_pts)

        pcd_tree = o3d.geometry.KDTreeFlann(pts_vis)
        overlaps = np.zeros([len(np.asarray(pts_vis.points))], dtype=bool)

        for idx in range(len(unproj_pts)):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(
                unproj_pts[idx, :3], 0.5, 1
            )
            if found:
                overlaps[idx_point] = True

        # check if any overlap have been found
        if True in np.unique(overlaps):

            # Define ICP parameters
            threshold = 0.02  # Maximum correspondence point-pair distance
            trans_init = np.identity(4)  # Initial transformation (identity matrix)
            max_iter = 1000  # Maximum number of ICP iterations
            overlap_base_pts_vis = draw_point_cloud(np.asarray(pts_vis.points)[overlaps[:]])


            reg_p2p = o3d.registration.registration_icp(
                unproj_pts_vis, overlap_base_pts_vis, threshold, trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=max_iter)
            )


            alignment_error = reg_p2p.inlier_rmse
            knn_distances.update({alignment_error: {"overlap": overlaps, "id": sample_id}})

    if len(knn_distances) < 1:
        print("[Info] No similarity have been found.")

    print("[Info] Sorting results")
    query_distances = [dis for dis in knn_distances.keys()]
    min_distance = min(query_distances)

    nearest_result = knn_distances[min_distance]
    result_scan_id = nearest_result["id"]
    result_overlap = nearest_result["overlap"]
    scene = Path(ROOT_DIRECTORY) / "scans" / result_scan_id / "{}_vh_clean_2.ply".format(result_scan_id)
    scene_ply= read_pc_from_ply(scene)
    scene_point_cloud = draw_point_cloud(scene_ply["points"])
    overlap_point_cloud = draw_point_cloud(np.asarray(scene_point_cloud.points)[result_overlap[:]])

    bbox = create_bounding_box(overlap_point_cloud)
    print("[Info] result shown from scene {}".format(result_scan_id))
    o3d.visualization.draw_geometries([bbox, scene_point_cloud], height=500, width=500)

    print("[Info] Writing into {}".format(save_dir))
    o3d.io.write_point_cloud(str(save_dir / "{}.ply".format(result_scan_id)), scene_point_cloud)
    np.save(save_dir / "{}_lines".format(result_scan_id), result_overlap)


if __name__ == "__main__":
    main()
