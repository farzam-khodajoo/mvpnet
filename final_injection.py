import os
import shutil
import glob
from pathlib import Path
import sys

PATH_3D_DATASET = "C:\\Users\\Arsham\\Desktop\\dataset\\scannet\\scans"
PATH_2D_DATASET = "C:\\Users\\Arsham\\Desktop\\dataset\\scannet\\2d_scannet"
PATH_RESIZED_DATASET = (
    "C:\\Users\\Arsham\\Desktop\\dataset\\scannet\\2d_scans_resize_160x120"
)

is_windows = sys.platform == "win32"

if is_windows:
    # changing to windows's path style
    PATH_3D_DATASET.replace("/", "\\")
    PATH_2D_DATASET.replace("/", "\\")
    PATH_RESIZED_DATASET.replace("/", "\\")


# check dataset exists
dataset_3d_exists = os.path.isdir(PATH_3D_DATASET)
dataset_2d_exists = os.path.isdir(PATH_2D_DATASET)
dataset_resized_exists = os.path.isdir(PATH_RESIZED_DATASET)

if not dataset_3d_exists or not dataset_2d_exists or not dataset_resized_exists:
    print("Terminate program!")
    exit()


def get_sub_folder(path: Path):
    path = Path(path)
    scene_id = str(path.absolute()).split(path.name)
    scene_id = Path(scene_id[0]).name
    return scene_id


resized_dataset_samples = glob.glob(os.path.join(PATH_RESIZED_DATASET, "*"))
all_3d_files = glob.glob(os.path.join(PATH_3D_DATASET, "*", "*"))
pose_2d_samples = glob.glob(os.path.join(PATH_2D_DATASET, "*", "pose"))
intrinsic_2d_samples = glob.glob(os.path.join(PATH_2D_DATASET, "*", "intrinsic"))

__COUNTER = 0

for resized_sample_dir in resized_dataset_samples:
    scan_id = Path(resized_sample_dir).name
    __COUNTER += 1
    print("[STATUS] Processing {} ({}/{})".format(scan_id, __COUNTER, len(resized_dataset_samples)))
    matchID = lambda path: get_sub_folder(path) == scan_id
    related_ply_files = list(filter(matchID, all_3d_files))
    related_pose_file = list(filter(matchID, pose_2d_samples))
    related_intrinsic_file = list(filter(matchID, intrinsic_2d_samples))

    if len(related_intrinsic_file) > 1 or len(related_intrinsic_file) > 1:
        print("[ERROR] multiple 2d folder for {}".format(scan_id))
        print(related_pose_file)
        print(related_intrinsic_file)

    if len(related_ply_files) < 1:
        print("[ERROR] no ply file found for {}".format(scan_id))
        print(related_ply_files)

    shutil.copytree(related_pose_file[0], Path(resized_sample_dir) / "pose")
    shutil.copytree(related_intrinsic_file[0], Path(resized_sample_dir) / "intrinsic")

    for file_3d in related_ply_files:
        shutil.copy(file_3d, resized_sample_dir)

    print("[STATUS] End processing {} ({}/{})".format(scan_id, __COUNTER, len(resized_dataset_samples)))

    #shutil.copytree()
