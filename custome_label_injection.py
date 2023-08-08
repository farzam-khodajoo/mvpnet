import os
import os.path as osp
import glob
import zipfile
from pathlib import Path

DATASET_DIRECTORY = "C:\\Users\\Arsham\\Desktop\\dataset\\scannet\\scans"
OUTPUT_DIRECTORY = "C:\\Users\\Arsham\\Desktop\\dataset\\scannet\\2d_scannet"


# changing to windows's path style
DATASET_DIRECTORY.replace("/", "\\")
OUTPUT_DIRECTORY.replace("/", "\\")

# check dataset exists
dataset_exists = osp.isdir(DATASET_DIRECTORY)
output_directory_exists = osp.isdir(OUTPUT_DIRECTORY)

if not dataset_exists or not output_directory_exists:
    print("Terminate program!")
    exit()

dataset_glob_paths = os.path.join(DATASET_DIRECTORY, "*", "*label.zip")
output_glob_path = os.path.join(OUTPUT_DIRECTORY, "*")

dataset_glob_paths = glob.glob(dataset_glob_paths)
output_glob_path = glob.glob(output_glob_path)


def get_sub_folder(path: Path):
    path = Path(path)
    scene_id = str(path.absolute()).split(path.name)
    scene_id = Path(scene_id[0]).name
    return scene_id


for idx, scene_path in enumerate(dataset_glob_paths):
    scene_path = Path(scene_path)
    scene_id = get_sub_folder(scene_path)
    corresponding_outputs = filter(
        lambda path: Path(path).name==scene_id, output_glob_path
    )
    
    corresponding_outputs = list(corresponding_outputs)
    if len(corresponding_outputs) > 1:
        print("[ERROR] found multiple match for one sample !")
        print(corresponding_outputs)
        exit()

    output_path = corresponding_outputs[0]
    # extract label.zip into output directory
    with zipfile.ZipFile(scene_path) as zip:
        zip.extractall(output_path)

    print("[STATUS] End Extracting {}/{}".format(idx + 1, len(dataset_glob_paths)))
