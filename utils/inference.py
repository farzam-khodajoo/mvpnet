import os
import os.path as osp
import sys
import logging
import time
import warnings
from pathlib import Path
from collections import OrderedDict

extend_path = os.getcwd()
if sys.platform == "win32":
    extend_path = extend_path.replace("/", "\\")

sys.path.append(extend_path)

from SETTING import ROOT_DIRECTORY, SCANNET_DIRECTORY

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from common.utils.checkpoint import CheckpointerV2
from common.utils.metric_logger import MetricLogger
from common.utils.torch_util import set_random_seed
from mvpnet.models.build import build_model_mvpnet_3d
from mvpnet.data.scannet_2d3d import ScanNet2D3DChunksTest
from mvpnet.evaluate_3d import Evaluator

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

def load_label_dictionary():

    # create label mapping
    all_class_ids = Path.cwd() / "mvpnet/data/meta_files/labelids.txt"
    label_mapping = load_class_mapping(all_class_ids)

    return label_mapping

def get_available_labels(labels_in_scene, label_dict):
    label_names = []
    for label_id in labels_in_scene:
        for (idx, name) in label_dict.items():
            if idx == label_id:
                label_names.append("{} [{}]".format(name, idx))

    return label_names


def inference_mvpnet(target_scene_id, output_dir='', run_name=''):

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from common.config import purge_cfg
    from mvpnet.config.mvpnet_3d import cfg
    config_file = Path.cwd() / "configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml"
    logging.info("Config file {} -> exists: {}".format(config_file, config_file.exists()))
    cfg.merge_from_file(str(config_file))
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)


    logger = logging.getLogger('mvpnet.test')

    # build mvpnet model
    model = build_model_mvpnet_3d(cfg)[0]
    model = model.cuda()

    # build checkpointer
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)
    # load last checkpoint
    checkpointer.load(None, resume=True)

    # build dataset
    k = 3
    num_views = 5
    min_nb_pts = 2048
    test_dataset = ScanNet2D3DChunksTest(cache_dir=str(Path(ROOT_DIRECTORY)/"pickles"),
                                         image_dir=str(Path(ROOT_DIRECTORY)/"scans_resize_160x120"),
                                         split="test",
                                         chunk_size=[1.5, 1.5],
                                         chunk_stride=0.5,
                                         chunk_thresh=1000,
                                         num_rgbd_frames=num_views,
                                         resize=(160, 120),
                                         image_normalizer=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                         k=k,
                                         to_tensor=True,
                                         )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=lambda x: x[0])

    # evaluator
    class_names = test_dataset.class_names
    evaluator = Evaluator(class_names)
    num_classes = len(class_names)
    submit_dir = None

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    set_random_seed(cfg.RNG_SEED)
    test_meters = MetricLogger(delimiter='  ')

    with torch.no_grad():
        start_time = time.time()
        start_time_scan = time.time()
        for scan_idx, data_dict_list in enumerate(test_dataloader):
            # fetch data
            scan_id = test_dataset.scan_ids[scan_idx]
            points = test_dataset.data[scan_idx]['points'].astype(np.float32)

            # prepare outputs
            num_points = len(points)
            pred_logit_whole_scene = np.zeros([num_points, num_classes], dtype=np.float32)
            num_pred_per_point = np.zeros(num_points, dtype=np.uint8)

            # iterate over chunks
            tic = time.time()
            for data_dict in data_dict_list:
                chunk_ind = data_dict.pop('chunk_ind')

                # padding for chunks with points less than min-nb-pts
                # It is required since farthest point sampling requires more points than centroids.
                chunk_points = data_dict['points']
                chunk_nb_pts = chunk_points.shape[1]  # note that already transposed
                if chunk_nb_pts < min_nb_pts:
                    print('Too sparse chunk in {} with {} points.'.format(scan_id, chunk_nb_pts))
                    pad = np.random.randint(chunk_nb_pts, size=min_nb_pts - chunk_nb_pts)
                    choice = np.hstack([np.arange(chunk_nb_pts), pad])
                    data_dict['points'] = data_dict['points'][:, choice]
                    data_dict['knn_indices'] = data_dict['knn_indices'][choice]

                data_batch = {k: torch.tensor([v]) for k, v in data_dict.items()}
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
                # forward 
                preds = model(data_batch)
                seg_logit = preds['seg_logit'].squeeze(0).cpu().numpy().T
                seg_logit = seg_logit[:len(chunk_ind)]
                # update
                pred_logit_whole_scene[chunk_ind] += seg_logit
                num_pred_per_point[chunk_ind] += 1

            pred_logit_whole_scene = pred_logit_whole_scene / np.maximum(num_pred_per_point[:, np.newaxis], 1)
            pred_label_whole_scene = np.argmax(pred_logit_whole_scene, axis=1)

            no_pred_mask = num_pred_per_point == 0
            no_pred_indices = np.nonzero(no_pred_mask)[0]
            if no_pred_indices.size > 0:
                logger.warning('{:s}: There are {:d} points without prediction.'.format(scan_id, no_pred_mask.sum()))
                pred_label_whole_scene[no_pred_indices] = num_classes

            return points, pred_label_whole_scene

