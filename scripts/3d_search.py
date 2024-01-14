VOXEL_SIZE = 0.0001
DEPTH_SCALE_FACTOR = 100.0

import sys
import os

is_windows = sys.platform == "win32"

extend_path = os.getcwd()
if is_windows:
    extend_path = extend_path.replace("/", "\\")

sys.path.append(extend_path)

from collections import OrderedDict
import os.path as osp
import csv
import logging
import argparse
from pathlib import Path
from glob import glob
import open3d as o3d
import numpy as np
from pathlib import Path
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
from prettytable import PrettyTable, ALL


from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color
from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY
from utils.general import (
    raise_path_error,
    Pickle
)
from utils.o3d import (
    ScannetDataset,
    PLYFormat,
    Common3D,
    unproject
)
from utils.inference import (
    load_class_mapping,
    get_available_labels,
    inference_mvpnet
)

logging.basicConfig(level=logging.INFO)

def main():
    global SCANNET_DIRECTORY
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=SCANNET_DIRECTORY,
        help="path to scans/ dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="sample which should be searched on dataset",
    )
    parser.add_argument(
        "--mvpnet",
        action="store_true",
        default=False,
        help="using mvpnet for fusion segmentation",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        default=None,
        help="label to be segmented from mvpnet",
    )
    parser.add_argument(
        "--icp-threshold",
        type=float,
        required=False,
        default=0.5,
        help="threshold parameter for ICP",
    )
    parser.add_argument(
        "--max-icp-iter",
        type=int,
        required=False,
        default=5_000,
        help="max iteration for ICP",
    )
    parser.add_argument(
        "--knn-radius",
        type=float,
        required=False,
        default=0.5,
        help="max iteration for ICP",
    )
    parser.add_argument(
        "--knn-nn",
        type=float,
        required=False,
        default=1,
        help="max iteration for ICP",
    )
    parser.add_argument(
        "--rmse-max",
        action="store_true",
        default=False,
        help="sort candidates based on maximum rmse",
    )

    args = parser.parse_args()

    if args.dataset is not None:
        if not Path(args.dataset).exists():
            logging.error("Dataset directory {} does not exists".format(args.dataset))
            exit()

        len_samples = glob(str(Path(args.dataset) / "*"))
        logging.info("Loading {} samples from dataset {}".format(len(len_samples), args.dataset))
        SCANNET_DIRECTORY = str(Path(args.dataset))

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

        pc = PLYFormat.read_pc_from_ply(scene_ply[0])
        pts_vis = draw_point_cloud(pc["points"]).voxel_down_sample(voxel_size=VOXEL_SIZE)

        bbox = np.load(bbox[0])
        overlap_point_cloud = draw_point_cloud(np.asarray(pts_vis.points)[bbox[:]])
        bbox = Common3D.create_bounding_box(overlap_point_cloud)
        logging.info("Loading scene")
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
        logging.error("color.jpg id and depth.jpg id mismatch !")
        logging.error(
            "color: {} --- depth: {}".format(color_sample_id, depth_sample_id)
        )
        exit()

    sample_pose = target_dir / "{}.txt".format(depth_sample_id)
    if not sample_pose.exists():
        raise_path_error("pose txt file", sample_pose)

    sample_intrinsic_depth = target_dir / "intrinsic_depth.txt"
    if not sample_intrinsic_depth.exists():
        raise_path_error("intrinsic file", sample_intrinsic_depth)

    # load image
    color_image = Image.open(sample_color)
    color = np.array(color_image)
    print("rgb", color.shape)

    # loading instrinsic data
    cam_matrix = np.loadtxt(sample_intrinsic_depth, dtype=np.float32)

    # load depth image
    depth_image = Image.open(sample_depth)
    depth = np.asarray(depth_image, dtype=np.float32) / DEPTH_SCALE_FACTOR

    # load pose data
    pose = np.loadtxt(sample_pose)

    # unporoject 2D to 3D
    unproj_pts = unproject(cam_matrix, depth)
    unproj_pts = pose[:3, :3].dot(unproj_pts[:, :3].T).T + pose[:3, 3]
    masked_unproj_pts = None


    logging.info("project cloud shape: {}".format(unproj_pts.shape))

    # visualize projection before proceed search
    logging.info("Loading pre-view of original projection")
    unproj_pts_vis = draw_point_cloud(unproj_pts)

    if masked_unproj_pts is not None:
        logging.info("Loading pre-view of masked projection")
        masked_unproj_pts_vis = draw_point_cloud(masked_unproj_pts)
        o3d.visualization.draw_geometries(
            [masked_unproj_pts_vis], height=500, width=500
        )
        logging.info("previous projection size: {}".format(unproj_pts.shape))
        unproj_pts = masked_unproj_pts
        logging.info("New masked projection size: {}".format(unproj_pts.shape))


    param_table = PrettyTable()
    param_table.field_names = ["Param", "value"]
    param_table.add_row(["KDTreeFlann radius", args.knn_radius])
    param_table.add_row(["KDTreeFlann nn", args.knn_nn])
    param_table.add_row(["ICP threshold", args.icp_threshold])
    param_table.add_row(["ICP maximum iteration", args.max_icp_iter])

    print(param_table)
    # container for knn distance results
    knn_distances = {}

    all_scenes = ScannetDataset.get_scenes(args.dataset)
    for idx, scene in enumerate(all_scenes):
        # check if folder exists
        if not scene.exists():
            raise_path_error("scene", scene)

        logging.info("Loading scene from {} ({}/{})".format(scene, idx+1, len(all_scenes)))

        sample_id = scene.name
        sample_ply_scene_path = scene / "{}_vh_clean_2.ply".format(sample_id)
        pc = PLYFormat.read_pc_from_ply(sample_ply_scene_path, return_color=True)
        pts_vis = draw_point_cloud(pc["points"])

        unproj_pts_vis = draw_point_cloud(unproj_pts)

        pcd_tree = o3d.geometry.KDTreeFlann(pts_vis)
        overlaps = np.zeros([len(np.asarray(pts_vis.points))], dtype=bool)

        for idx in tqdm(range(len(unproj_pts))):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(
                unproj_pts[idx, :3], args.knn_radius, args.knn_nn
            )
            if found:
                overlaps[idx_point] = True

        # check if any overlap have been found
        if True in np.unique(overlaps):
            # Define ICP parameters
            threshold = 0.5  # Maximum correspondence point-pair distance
            trans_init = np.identity(4)  # Initial transformation (identity matrix)
            max_iter = 5_000  # Maximum number of ICP iterations
            overlap_base_pts_vis = draw_point_cloud(
                np.asarray(pts_vis.points)[overlaps[:]]
            )

            reg_p2p = o3d.registration.registration_icp(
                unproj_pts_vis,
                overlap_base_pts_vis,
                args.icp_threshold,
                trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=args.max_icp_iter),
            )

            alignment_error = reg_p2p.inlier_rmse

            knn_distances.update(
                {alignment_error: {"overlap": overlaps, "id": sample_id}}
            )

    if len(knn_distances) < 1:
        logging.warning("No similarity have been found.")
        exit()

    logging.info("Sorting results")
    query_distances = [dis for dis in knn_distances.keys()]
    min_distance = max(query_distances) if args.rmse_max else min(query_distances)

    nearest_result = knn_distances[min_distance]
    result_scan_id = nearest_result["id"]
    result_overlap = nearest_result["overlap"]
    logging.info("Found #1 first candidate {} with RMSE of {}".format(result_scan_id, min_distance))


    if len(knn_distances) >= 2:

        query_distances.sort(reverse=True if args.rmse_max else False)
        second_min_distance = query_distances
        second_min_distance = second_min_distance[1] # [0] will be first

        if query_distances[0] != min_distance:
            logging.error("Result sorting mismatching")
            exit()

        second_nearest_result = knn_distances[second_min_distance]
        second_scan_id = second_nearest_result["id"]
        logging.info("Found #2 second candidate {} with RMSE of {}".format(second_scan_id, second_min_distance))


    if args.dataset is not None:
        scene = (
            Path(args.dataset)
            / result_scan_id
            / "{}_vh_clean_2.ply".format(result_scan_id)
        )

    else:
        scene = (
            Path(SCANNET_DIRECTORY)
            / result_scan_id
            / "{}_vh_clean_2.ply".format(result_scan_id)
        )
    

    target_fixed_color = [0.7, 0.2, 0.1]
    scene_ply = PLYFormat.read_pc_from_ply(scene, return_color=True)
    scene_colors = scene_ply["colors"] / 255.
    scene_point_cloud = draw_point_cloud(scene_ply["points"], colors=scene_colors)
    overlap_point_cloud = draw_point_cloud(
        np.asarray(scene_point_cloud.points)[result_overlap[:]], colors=target_fixed_color
    )

    bbox: o3d.geometry.LineSet = Common3D.create_bounding_box(overlap_point_cloud)
    scene_with_fixed_overlap = Common3D.set_color_for_overlaps_in_scene(
            scene_pts=scene_point_cloud,
            scene_colors=scene_colors,
            overlap_pts=overlap_point_cloud,
            color=target_fixed_color
        )

    if not args.mvpnet:
        logging.info("Result shown from scene {}".format(result_scan_id))
        o3d.visualization.draw_geometries([bbox, scene_with_fixed_overlap], height=500, width=500)


    logging.info("Writing into {}".format(save_dir))
    o3d.io.write_point_cloud(
        str(save_dir / "{}.ply".format(result_scan_id)), scene_point_cloud
    )
    np.save(save_dir / "{}_lines".format(result_scan_id), result_overlap)

    # saving rmse results into report/rmse.csv
    logging.info("Getting RMSE report..")
    import pandas as pd

    rmse_report_context = {
        "space_id": [],
        "rmse_score": []
    }

    for rmse_score in knn_distances.keys():
        room_name = knn_distances[rmse_score]["id"]
        rmse_report_context["space_id"].append(room_name)
        rmse_report_context["rmse_score"].append(rmse_score)


    rmse_df = pd.DataFrame(rmse_report_context)
    print(rmse_df.to_markdown())

    rmse_report_save_destination = save_dir / "rmse.csv"
    logging.info("Sasving RMSE report into {}".format(rmse_report_save_destination))
    rmse_df.to_csv(rmse_report_save_destination)

    # save center xyz into file
    with open(save_dir / "position.txt", "w") as file:
        centroid = overlap_point_cloud.get_center()
        logging.info("Saving {} coordinate into {}".format(centroid, save_dir / "position.txt"))
        file.write(f"{centroid[0]}\n{centroid[1]}\n{centroid[2]}")


    if args.mvpnet:
        from mvpnet.utils.visualize import visualize_labels
        points, prediction_labels = inference_mvpnet()
        all_class_ids = Path.cwd() / "mvpnet/data/meta_files/labelids.txt"
        label_dict = load_class_mapping(all_class_ids)   
        labels_in_scene = np.unique(prediction_labels)

        all_label_names = get_available_labels(labels_in_scene, label_dict)
        points_insize_bbox = np.all((overlap_point_cloud.get_min_bound() <= scene_point_cloud.points)
                                    & (scene_point_cloud.points <= overlap_point_cloud.get_max_bound()), axis=1)

        prediction_labels_inside_bbox = prediction_labels[points_insize_bbox]
        bbox_label_names = np.unique(prediction_labels_inside_bbox)
        bbox_label_names = get_available_labels(bbox_label_names, label_dict)

        label_table = PrettyTable()
        label_table.hrules = ALL
        label_table.field_names = ["Name", "Labels"]
        label_table.add_row(["All scene", all_label_names])
        label_table.add_row(["inside bounding box", bbox_label_names])

        print(label_table)
        #visualize_labels(points, prediction_labels)

        if args.label:
            index = list(label_dict.keys()).index(int(args.label))
            target_indices = np.where(prediction_labels_inside_bbox == index)
            color_space = label2color(prediction_labels)[target_indices]
            scene_colors[target_indices] = color_space
            scene_point_cloud.colors = o3d.utility.Vector3dVector(scene_colors)
            o3d.visualization.draw_geometries([bbox, scene_point_cloud])


        else:
            scene_with_segmented_overlap = Common3D.set_color_for_overlaps_in_scene(
                    scene_pts=scene_point_cloud,
                    scene_colors=scene_colors,
                    overlap_pts=overlap_point_cloud,
                    color=label2color(prediction_labels)
            )
            
            o3d.visualization.draw_geometries([bbox, scene_with_segmented_overlap], height=500, width=500)
        

if __name__ == "__main__":
    main()