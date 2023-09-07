import os.path as osp
import logging
import pickle

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import open3d as o3d
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

_CUR_DIR = osp.dirname(__file__)
_META_DIR = osp.abspath(osp.join(_CUR_DIR, "meta_files"))

from mvpnet.data.scannet_2d import load_class_mapping, read_label_mapping
from mvpnet.utils.chunk_util import scene2chunks_legacy

from sys import getsizeof

def scene2chunks_legacy(
    points, chunk_size, stride, thresh=1000, margin=(0.2, 0.2), return_bbox=False
):
    """Split the whole scene into chunks based on the original PointNet++ implementation.
    Only slide chunks on the xy-plane

    Args:
        points (np.ndarray): (num_points, 3)
        chunk_size (2-tuple): size of chunk
        stride (float): stride of chunk
        thresh (int): minimum number of points in a qualified chunk
        margin (2-tuple): margin of chunk
        return_bbox (bool): whether to return bounding boxes

    Returns:
        chunk_indices (list of np.ndarray)
        chunk_bboxes (list of np.ndarray, optional): each bbox is (x1, y1, z1, x2, y2, z2)

    """
    chunk_size = np.asarray(chunk_size)
    margin = np.asarray(margin)

    coord_max = np.max(points, axis=0)  # max x,y,z
    coord_min = np.min(points, axis=0)  # min x,y,z
    limit = coord_max - coord_min
    # get the corner of chunks.
    num_chunks = np.ceil((limit[:2] - chunk_size) / stride).astype(int) + 1
    corner_list = []
    for i in range(num_chunks[0]):
        for j in range(num_chunks[1]):
            corner_list.append((coord_min[0] + i * stride, coord_min[1] + j * stride))

    xy = points[:, :2]
    chunk_indices = []
    chunk_bboxes = []
    for corner in corner_list:
        corner = np.asarray(corner)
        mask = np.all(np.logical_and(xy >= corner, xy <= corner + chunk_size), axis=1)
        # discard unqualified chunks
        if np.sum(mask) < thresh:
            continue
        mask = np.all(
            np.logical_and(xy >= corner - margin, xy <= corner + chunk_size + margin),
            axis=1,
        )
        indices = np.nonzero(mask)[0]
        chunk_indices.append(indices)
        if return_bbox:
            chunk = points[indices]
            bbox = np.hstack(
                [
                    corner - margin,
                    chunk.min(0)[2],
                    corner + chunk_size + margin,
                    chunk.max(0)[2],
                ]
            )
            chunk_bboxes.append(bbox)
    if return_bbox:
        return chunk_indices, chunk_bboxes
    else:
        return chunk_indices


def select_frames(rgbd_overlap, num_rgbd_frames):
    selected_frames = []
    # make a copy to avoid modifying input
    rgbd_overlap = rgbd_overlap.copy()
    for i in range(num_rgbd_frames):
        # choose the RGBD frame with the maximum overlap (measured in num basepoints)
        frame_idx = rgbd_overlap.sum(0).argmax()
        selected_frames.append(frame_idx)
        # set all points covered by this frame to invalid
        rgbd_overlap[rgbd_overlap[:, frame_idx]] = False
    return selected_frames


def depth2xyz(cam_matrix, depth):
    # create xyz coordinates from image position
    v, u = np.indices(depth.shape)
    u, v = u.ravel(), v.ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    xyz = (np.linalg.inv(cam_matrix[:3, :3]).dot(uv1_points.T) * depth.ravel()).T
    return xyz


def compute_rgbd_knn():
    pass


def get_rgbd_data(
    chunk_points,
    sample_color,
    sample_depth,
    sample_pose,
    cam_matrix,
    num_rgbd_frames,
    resize=(160, 120),
    depth_scale_factor: float = 1000.0,
):
    image = sample_color
    depth = sample_depth
    pose = sample_pose

    image_list = []
    image_xyz_list = []
    image_mask_list = []

    # resize
    if resize:
        if not image.size == resize:
            # check if we do not enlarge downsized images
            assert image.size[0] > resize[0] and image.size[1] > resize[1]
            image = image.resize(resize, Image.BILINEAR)
            depth = depth.resize(resize, Image.NEAREST)

    # normalize image
    image = np.asarray(image, dtype=np.float32) / 255.0

    image_list.append(image)
    # rescale depth
    depth = np.asarray(depth, dtype=np.float32) / depth_scale_factor
    # inverse perspective transformation
    image_xyz = depth2xyz(cam_matrix, depth)  # (h * w, 3)
    # find valid depth
    image_mask = image_xyz[:, 2] > 0  # (h * w)
    # camera -> world
    image_xyz = np.matmul(image_xyz, pose[:3, :3].T) + pose[:3, 3]

    if not np.any(image_mask):
        logging.warning("Invalid depth map ! (inf)")

    image_xyz_list.append(image_xyz)
    image_mask_list.append(image_mask)

    # post-process, especially for horizontal flip
    image_ind_list = []
    for i in range(num_rgbd_frames):
        h, w, _ = image_list[i].shape
        # reshape
        image_xyz_list[i] = image_xyz_list[i].reshape([h, w, 3])
        image_mask_list[i] = image_mask_list[i].reshape([h, w])

        image_mask = image_mask_list[i]
        image_ind = np.nonzero(image_mask.ravel())[0]

        if image_ind.size > 0:
            image_ind_list.append(image_ind + i * h * w)
        else:
            image_ind_list.append([])

        images = np.stack(image_list, axis=0)  # (nv, h, w, 3)
        image_xyz_valid = np.concatenate(
            [
                image_xyz[image_mask]
                for image_xyz, image_mask in zip(image_xyz_list, image_mask_list)
            ],
            axis=0,
        )

        image_ind_all = np.hstack(image_ind_list)  # (n_valid,)

        # Find k-nn in dense point clouds for each sparse point
        nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(
            image_xyz_valid
        )
        _, knn_indices = nbrs.kneighbors(chunk_points)  # (nc, 3)
        # remap to pixel index
        knn_indices = image_ind_all[knn_indices]

        out_dict = {
            "images": images.astype(np.float32, copy=False),
            "image_xyz": np.stack(image_xyz_list, axis=0).astype(
                np.float32, copy=False
            ),  # (nv, h, w, 3)
            "image_mask": np.stack(image_mask_list, axis=0).astype(
                np.bool_, copy=False
            ),  # (nv, h, w)
            "knn_indices": knn_indices.astype(np.int64, copy=False),  # (nc, 3)
        }
        return out_dict


def get_input_batch_sample(
    points,
    color,
    depth,
    pose,
    cam_matrix,
    chunk_size=(2, 2),
    chunk_thresh=1000,
    chunk_stride=0.5,
    chunk_margin=(10, 10),
    depth_scale_factor=1000.0
):
    nb_pts=-1    
    num_rgbd_frames = 1
    points = points.astype(np.float32)
    chunk_indices, chunk_boxes = scene2chunks_legacy(
        points,
        chunk_size=np.array(chunk_size, dtype=np.float32),
        stride=chunk_stride,
        thresh=chunk_thresh,
        margin=chunk_margin,
        return_bbox=True,
    )

    out_dict = {
        "points": points
    }

    out_dict_rgbd = get_rgbd_data(
        chunk_points=points,
        sample_color=color,
        sample_depth=depth,
        sample_pose=pose,
        cam_matrix=cam_matrix,
        num_rgbd_frames=num_rgbd_frames,
        depth_scale_factor=depth_scale_factor
    )

    out_dict.update(out_dict_rgbd)


    # to_tensor
    points = points.T  # (3, nc)
    if "images" in out_dict:
        out_dict["images"] = np.moveaxis(
            out_dict["images"], -1, 1
        )  # (nv, 3, h, w)

    out_dict["points"] = points

    return out_dict