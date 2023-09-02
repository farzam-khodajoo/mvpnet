from collections import OrderedDict
import os.path as osp
import csv
import logging
from typing import List
import argparse
import pickle
import torch
from pathlib import Path
from glob import glob
from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY
from mvpnet.utils.o3d_util import draw_point_cloud
from mvpnet.utils.visualize import label2color
from mvpnet.data.scannet_2d3d import ScanNet2D3DChunks
from common.utils.checkpoint import CheckpointerV2
from mvpnet.config.mvpnet_3d import cfg
from custome_sampler import get_input_batch_sample

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm


DEPTH_SCALE_FACTOR = 100.0

logging.basicConfig(level=logging.INFO)


def raise_path_error(title, path=""):
    logging.error("{} {}".format(title, path))
    exit()


def load_from_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_into_pickle(object, path: str):
    with open(path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def load_class_mapping(filename):
    id_to_class = OrderedDict()
    with open(filename, "r") as f:
        for line in f.readlines():
            class_id, class_name = line.rstrip().split("\t")
            id_to_class[int(class_id)] = class_name
    return id_to_class


def read_label_mapping(
    filename, label_from="raw_category", label_to="nyu40id", as_int=False
):
    """Read the mapping of labels from the given tsv"""
    assert osp.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            key = row[label_from]
            if as_int:
                key = int(key)
            mapping[key] = int(row[label_to])
    return mapping


def load_model(config):
    # the reason this dependencies are imported right here
    # is that model side requires compile CUDA libraries
    # which is impossible on non-nvidia device
    # so we import only when model is required
    from mvpnet.models.build import build_model_mvpnet_3d

    cfg.merge_from_file(str(config))
    model = build_model_mvpnet_3d(cfg)[0]
    model = model.cuda()
    model.eval()

    return model


def get_prediction_labels(predict):
    seg_logit = predict["seg_logit"].squeeze(0).cpu().detach().numpy().T

    return seg_logit


def predict_segmentation(model, data_batch: dict):
    # convert numpy into torch.tensor
    data_batch = {k: torch.tensor([v]) for k, v in data_batch.items()}
    # send to CUDA memory
    data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

    predict = model(data_batch)
    # send back to CPU
    predict = {k: v.cpu() for k, v in predict.items()}

    return get_prediction_labels(predict)


def get_prediction_segmentation_from_pickle(path):
    logits = load_from_pickle(path)
    return logits


def load_cache(path):
    if not Path(path).exists():
        raise_path_error("cache file", path)

    with open(path, "rb") as cache:
        print("[Info] Loading cache file")
        cache_data = pickle.load(cache)
        return cache_data


def search_scanid(cache_data, scanid):
    id_query = [d for d in cache_data if d["scan_id"] == scanid]
    if len(id_query) == 0:
        raise_path_error("query scan id", scanid)

    return id_query[0]


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
        np.array([[1, 0, 0] for _ in range(len(lines))])
    )
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
        "--mvpnet",
        action="store_true",
        default=False,
        help="using mvpnet for fusion segmentation",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        required=False,
        default=None,
        help="path to cache .pickle file",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=False,
        default=None,
        help="label to be segmented from mvpnet",
    )
    parser.add_argument(
        "--save-segmentation-as-pickle",
        type=str,
        required=False,
        default=None,
        help="save model logit into pickle",
    )
    parser.add_argument(
        "--compare-mvpnet-labels",
        action="store_true",
        required=False,
        default=None,
        help="log difference between predicted labels and real one",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        required=False,
        default=None,
        help="avoid searching 3D search",
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

    if args.mvpnet:
        # check if config file exists
        model_config = (
            Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"
        )
        if not model_config.exists():
            raise_path_error("pn2ssg config file", model_config)

        # pre-process data
        logging.info("Loading batch sample")
        data = get_input_batch_sample(
            points=unproj_pts,
            color=color_image,
            depth=depth_image,
            pose=pose,
            cam_matrix=cam_matrix,
            depth_scale_factor=1000.0,
        )

        # create label mapping
        all_class_ids = Path.cwd() / "mvpnet/data/meta_files/labelids.txt"
        if not all_class_ids.exists():
            raise_path_error("file", all_class_ids)

        labels_combined_path = (
            Path.cwd() / "mvpnet/data/meta_files/scannetv2-labels.combined.tsv"
        )
        if not labels_combined_path.exists():
            raise_path_error("file", labels_combined_path)

        raw_to_nyu40_mapping = read_label_mapping(labels_combined_path,
                                                       label_from='id', label_to='nyu40id', as_int=True)

        raw_to_nyu40 = np.zeros(max(raw_to_nyu40_mapping.keys()) + 1, dtype=np.int64)  

        for key, value in raw_to_nyu40_mapping.items():
            raw_to_nyu40[key] = value

        label_mapping = load_class_mapping(all_class_ids)
        assert len(label_mapping) == 20
        nyu40_to_scannet = np.full(shape=41, fill_value=-100, dtype=np.int64)
        nyu40_to_scannet[list(label_mapping.keys())] = np.arange(len(label_mapping))
        scannet_to_nyu40 = np.array(list(label_mapping.keys()) + [0], dtype=np.int64)
        raw_to_scannet = nyu40_to_scannet[raw_to_nyu40]
        label_map_names = tuple(label_mapping.values())
        print(label_map_names)
        print(nyu40_to_scannet)
        print(len(nyu40_to_scannet))
        label_map_ids = [id_ for id_, classname in label_mapping.items()]

        if args.pickle is not None:
            logging.info("Loading segmentation from pickle")
            segmentation = get_prediction_segmentation_from_pickle(args.pickle)

        else:
            logging.info("Loading model")
            model = load_model(config=model_config)

            logging.info("Loading weights")
            output_directory = (
                Path.cwd() / "outputs/scannet/mvpnet_3d_unet_resnet34_pn2ssg"
            )

            if not output_directory.exists() or not output_directory.is_dir():
                logging.error(
                    "model checkpoint direcotry {} does not exists".format(
                        output_directory
                    )
                )
                exit()

            logging.info("Loading weights from checkpointer")
            checkpointer = CheckpointerV2(model, save_dir=output_directory)
            if not checkpointer.has_checkpoint():
                logging.error("No checkpoint has been found")

            # this will resume and load last checkpoint
            checkpointer.load(None, resume=True)

            logging.info("Processing segmentation")
            segmentation = predict_segmentation(model=model, data_batch=data)

        # save model output into pickle
        if args.save_segmentation_as_pickle is not None:
            if args.pickle is not None:
                logging.error("avoid re-saving pickle")
                exit()

            logging.info(
                "Save segmentation output into {}".format(
                    args.save_segmentation_as_pickle
                )
            )
            save_into_pickle(segmentation, args.save_segmentation_as_pickle)

        scene_predicted_labels = np.argmax(segmentation, axis=1)

        predicted_labels = np.unique(scene_predicted_labels)
        logging.info("Label IDs found in prediction {}".format(predicted_labels))
        predicted_classnames = [label_map_names[int(idx)] for idx in predicted_labels]
        logging.info("Labels found in prediction: {}".format(predicted_classnames))

        # pop window and show differne between prediction and labels
        if args.compare_mvpnet_labels is not None:
            labels = Path(args.target) / "addition" / "{}.png".format(depth_sample_id)
            if not labels.exists():
                raise_path_error("label file", labels)

            # load labeled image and segment with only valid labels
            label_image = Image.open(labels)
            label_np = np.array(label_image)
            bad_label_ind = np.logical_or(label_np < 0, label_np > 40)
            label_np[bad_label_ind] = 0
            segmentaion_labels = nyu40_to_scannet[label_np]

            logging.info("Plotting label images")
            plt.figure(figsize=(7, 4))
            #label image including invalid images
            plt.subplot(121)
            plt.imshow(label_np)
            plt.title("Original")

            plt.subplot(122)
            plt.imshow(segmentaion_labels)
            plt.title("Valid only (exclude -100)")

            plt.tight_layout()
            plt.show()

            mvpnet_segmented_pt = draw_point_cloud(
                unproj_pts, label2color(scene_predicted_labels)
            )

            logging.info("Loading segmented view from mvpnet")
            o3d.visualization.draw_geometries(
                [mvpnet_segmented_pt], height=500, width=500
            )

        if args.label is not None:
            try:
                index_of_classname = label_map_names.index(args.label)
            except ValueError:
                logging.error(
                    "Classname {} not found in {}".format(args.label, label_map_names)
                )
                exit()

            logging.info(
                "Slicing projection with mask label '{}' and index of {}".format(
                    args.label, index_of_classname
                )
            )
            choosen_label = index_of_classname
            label_mask = np.where(scene_predicted_labels == choosen_label)
            masked_unproj_pts = unproj_pts[label_mask]

        if args.no_search:
            logging.info("Exit program after --no-search")
            exit()

    # visualize projection before proceed search
    logging.info("Loading pre-view of original projection")
    unproj_pts_vis = draw_point_cloud(unproj_pts)
    o3d.visualization.draw_geometries([unproj_pts_vis], height=500, width=500)

    if masked_unproj_pts is not None:
        logging.info("Loading pre-view of masked projection")
        masked_unproj_pts_vis = draw_point_cloud(masked_unproj_pts)
        o3d.visualization.draw_geometries(
            [masked_unproj_pts_vis], height=500, width=500
        )
        unproj_pts = masked_unproj_pts

    logging.info("Searching scenes")
    # container for knn distance results
    knn_distances = {}

    for scene in tqdm(get_scenes()):
        # check if folder exists
        if not scene.exists():
            raise_path_error("scene", scene)

        sample_id = scene.name
        sample_ply_scene_path = scene / "{}_vh_clean_2.ply".format(sample_id)
        pc = read_pc_from_ply(sample_ply_scene_path, return_color=True)
        pts_vis = draw_point_cloud(pc["points"])

        unproj_pts_vis = draw_point_cloud(unproj_pts)

        pcd_tree = o3d.geometry.KDTreeFlann(pts_vis)
        overlaps = np.zeros([len(np.asarray(pts_vis.points))], dtype=bool)

        for idx in tqdm(range(len(unproj_pts))):
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
            overlap_base_pts_vis = draw_point_cloud(
                np.asarray(pts_vis.points)[overlaps[:]]
            )

            reg_p2p = o3d.registration.registration_icp(
                unproj_pts_vis,
                overlap_base_pts_vis,
                threshold,
                trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=max_iter),
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
    min_distance = min(query_distances)

    nearest_result = knn_distances[min_distance]
    result_scan_id = nearest_result["id"]
    result_overlap = nearest_result["overlap"]
    scene = (
        Path(SCANNET_DIRECTORY)
        / result_scan_id
        / "{}_vh_clean_2.ply".format(result_scan_id)
    )
    scene_ply = read_pc_from_ply(scene)
    scene_point_cloud = draw_point_cloud(scene_ply["points"])
    overlap_point_cloud = draw_point_cloud(
        np.asarray(scene_point_cloud.points)[result_overlap[:]]
    )

    bbox = create_bounding_box(overlap_point_cloud)
    logging.info("Result shown from scene {}".format(result_scan_id))
    o3d.visualization.draw_geometries([bbox, scene_point_cloud], height=500, width=500)

    logging.info("Writing into {}".format(save_dir))
    o3d.io.write_point_cloud(
        str(save_dir / "{}.ply".format(result_scan_id)), scene_point_cloud
    )
    np.save(save_dir / "{}_lines".format(result_scan_id), result_overlap)


if __name__ == "__main__":
    main()
