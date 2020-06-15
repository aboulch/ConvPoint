import numpy as np
import argparse
import os
import semantic3D_utils.lib.python.semantic3D as Sem3D


parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', '-s', help='Path to data folder')
parser.add_argument("--savedir", type=str, default="./semantic3d_processed")
parser.add_argument("--voxel", type=float, default=0.1)
parser.add_argument("--checkfiles", action="store_true")
args = parser.parse_args()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC
def wred(str):
    return bcolors.FAIL+str+bcolors.ENDC


filelist_train=[
        "bildstein_station1_xyz_intensity_rgb",
        "bildstein_station3_xyz_intensity_rgb",
        "bildstein_station5_xyz_intensity_rgb",
        "domfountain_station1_xyz_intensity_rgb",
        "domfountain_station2_xyz_intensity_rgb",
        "domfountain_station3_xyz_intensity_rgb",
        "neugasse_station1_xyz_intensity_rgb",
        "sg27_station1_intensity_rgb",
        "sg27_station2_intensity_rgb",
        "sg27_station4_intensity_rgb",
        "sg27_station5_intensity_rgb",
        "sg27_station9_intensity_rgb",
        "sg28_station4_intensity_rgb",
        "untermaederbrunnen_station1_xyz_intensity_rgb",
        "untermaederbrunnen_station3_xyz_intensity_rgb",
    ]

filelist_test = [
        "birdfountain_station1_xyz_intensity_rgb",
        "castleblatten_station1_intensity_rgb",
        "castleblatten_station5_xyz_intensity_rgb",
        "marketplacefeldkirch_station1_intensity_rgb",
        "marketplacefeldkirch_station4_intensity_rgb",
        "marketplacefeldkirch_station7_intensity_rgb",
        "sg27_station10_intensity_rgb",
        "sg27_station3_intensity_rgb",
        "sg27_station6_intensity_rgb",
        "sg27_station8_intensity_rgb",
        "sg28_station2_intensity_rgb",
        "sg28_station5_xyz_intensity_rgb",
        "stgallencathedral_station1_intensity_rgb",
        "stgallencathedral_station3_intensity_rgb",
        "stgallencathedral_station6_intensity_rgb",
        ]

print("Creating train directories...", end="", flush=True)
savedir = os.path.join(args.savedir, "train", "pointcloud_txt")
os.makedirs(savedir, exist_ok=True)
savedir_numpy = os.path.join(args.savedir, "train", "pointcloud")
os.makedirs(savedir_numpy, exist_ok=True)
print("done")


print("Generating train files...")
for filename in filelist_train:
    print(wblue(filename))
    
    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    if os.path.exists(os.path.join(args.rootdir, "TRAIN", filename_txt)):
        if os.path.exists(os.path.join(args.rootdir, "TRAIN", filename_labels)):
            
            #if checkfiles flag, do not compute points
            if args.checkfiles: 
                continue

            # load file and voxelize
            Sem3D.semantic3d_load_from_txt_voxel_labels(os.path.join(args.rootdir, "TRAIN", filename_txt),
                                                        os.path.join(args.rootdir, "TRAIN", filename_labels),
                                                        os.path.join(savedir, filename+"_voxels.txt"),
                                                        args.voxel
                                                        )
            
            # save the numpy data
            np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))
        else:
            print(wred(f'Error -- label file does not exists: {os.path.join(args.rootdir, "TRAIN", filename_labels)}'))
    else:
        print(wred(f'Error -- points file does not exists: {os.path.join(args.rootdir, "TRAIN", filename_txt)}'))

print("Done")


print("Creating test directories...", end="", flush=True)
savedir = os.path.join(args.savedir, "test", "pointcloud_txt")
os.makedirs(savedir, exist_ok=True)
savedir_numpy = os.path.join(args.savedir, "test", "pointcloud")
os.makedirs(savedir_numpy, exist_ok=True)
print("done")


print("Generating test files...")
for filename in filelist_test:
    print(wgreen(filename))
    
    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    if os.path.exists(os.path.join(args.rootdir, "TEST", filename_txt)):
            
        #if checkfiles flag, do not compute points
        if args.checkfiles: 
            continue

        # load file and voxelize
        Sem3D.semantic3d_load_from_txt_voxel(os.path.join(args.rootdir, "TEST", filename_txt),
                                                    os.path.join(savedir, filename+"_voxels.txt"),
                                                    args.voxel
                                                    )
        
        # save the numpy data
        np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))
    else:
        print(wred(f'Error -- point file does not exists: {os.path.join(args.rootdir, "TEST", filename_txt)}'))

print("Done")