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
from mvpnet.config.mvpnet_3d import cfg

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
from prettytable import PrettyTable, ALL

logging.basicConfig(level=logging.INFO)


def raise_path_error(title, path=""):
    logging.error("{} {}".format(title, path))
    exit()

class Pickle:
    """
    All common pickle utilities used cross the project
    """

    @staticmethod
    def load_from_pickle(path: str):
        if not Path(path).exists():
            raise_path_error("pickle file", path)

        with open(path, "rb") as handle:
            return pickle.load(handle)
        
    @staticmethod
    def save_into_pickle(object, path: str):
        with open(path, "wb") as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
