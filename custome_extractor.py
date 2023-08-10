import glob
import os
import os.path as osp
import subprocess
import multiprocessing as mp
import sys

# 1. Place this file in master directory i.e sample place as enviroment.yaml
# 2. create python version 2.7 enviroment:
# -- conda create --name py27 python=2.7
# -- conda activate py27
# 3. Install required dependencies (python v2.7):
# -- pip install numpy  imageio==1.4 opencv-python==4.2.0.32
# versions were selected manually to support python <= 2.x

#NOTE Edit this:
DATASET_DIRECTORY = "C:\Users\Arsham\Desktop\dataset\scannet\scans" # Replace with your own dataset
OUTPUT_DIRECTORY = "C:\Users\Arsham\Desktop\dataset\scannet" # Replace with your own prefered directory
__SUB_PREFIX = "2d_scannet"
MULTI_PROCESSING_CORE = 1

# Python 2 script
MODULE_PATH = "%s\\mvpnet\\data\\preprocess\\SensReader\\reader.py" % os.getcwd()

is_windows = sys.platform == "win32"

if not is_windows:
    MODULE_PATH = MODULE_PATH.replace("\\", "/")

if is_windows:
    # changing to windows's path style
    DATASET_DIRECTORY = DATASET_DIRECTORY.replace("/", "\\")
    OUTPUT_DIRECTORY = OUTPUT_DIRECTORY.replace("/", "\\")


# check dataset exists
dataset_exists = osp.isdir(DATASET_DIRECTORY)
output_directory_exists = osp.isdir(OUTPUT_DIRECTORY)
module_exists = osp.isfile(MODULE_PATH)

print "[INFO] check dataset folder exists [ %s ] -> %s" % (DATASET_DIRECTORY, dataset_exists)
print "[INFO] check output folder exists [ %s ] -> %s" % (OUTPUT_DIRECTORY, output_directory_exists)
print "[INFO] check reader.py module exists [ %s ] -> %s" % (MODULE_PATH, module_exists)

if not dataset_exists or not output_directory_exists or not module_exists:
    print 'Terminate program!'
    exit()

# query all .sens file cross all samples
glob_paths = os.path.join(DATASET_DIRECTORY, '*', '*.sens')
glob_paths = glob.glob(glob_paths)

print "[INFO] number of samples found in dataset -> %s" % len(glob_paths)

if len(glob_paths) > 0:
    print "sample file path: %s" % glob_paths[0]

_COUNTER = 0

def extract(a):
    global _COUNTER
    i, sens_path = a
    rest, sens_filename = os.path.split(sens_path)
    scan_id = os.path.split(rest)[1]
    output_path = os.path.join(OUTPUT_DIRECTORY, __SUB_PREFIX, scan_id)

    if not os.path.exists(output_path):
        print 'creating folder [ %s ]' % output_path
        os.makedirs(output_path)

    print '[STATUS] Processing file %s/%s: %s ' % (i + 1, len(glob_paths), sens_filename)
    process = subprocess.Popen(['python', MODULE_PATH,
                                '--filename', sens_path,
                                '--output_path', output_path,
                                '--export_depth_images',
                                '--export_color_images',
                                '--export_poses',
                                '--export_intrinsics']
                               )
    process.wait()
    print '[STATUS] End processing %s/%s' % (_COUNTER+1, len(glob_paths))
    _COUNTER +=1


# with multiprocessing
samples = []
for i in range(len(glob_paths)):
    samples.append((i, glob_paths[i]))

print '[INFO] Start multi-processing'

if __name__ == "__main__":
    p = mp.Pool(MULTI_PROCESSING_CORE)
    p.map(extract, samples, chunksize=1)
    p.close()
    p.join()