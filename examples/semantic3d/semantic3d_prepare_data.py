import numpy as np
import argparse
import os
import semantic3D_utils.lib.python.semantic3D as Sem3D


parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', '-s', help='Path to data folder')
parser.add_argument("--savedir", type=str, default="./semantic3d_processed")
parser.add_argument("--voxel", type=float, default=0.1)
args = parser.parse_args()

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

print("Generating train files...")
for filename in filelist_train:
    print(filename)
    
    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    # load file and voxelize
    savedir = os.path.join(args.savedir, "train", "pointcloud_txt")
    os.makedirs(savedir, exist_ok=True)
    Sem3D.semantic3d_load_from_txt_voxel_labels(os.path.join(args.rootdir, "TRAIN", filename_txt),
                                                os.path.join(args.rootdir, "TRAIN", filename_labels),
                                                os.path.join(savedir, filename+"_voxels.txt"),
                                                args.voxel
                                                )
    
    # save the numpy data
    savedir_numpy = os.path.join(args.savedir, "train", "pointcloud")
    os.makedirs(savedir_numpy, exist_ok=True)
    np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))

print("Done")

print("Generating test files...")
for filename in filelist_test:
    print(filename)
    
    filename_txt = filename+".txt"
    filename_labels = filename+".labels"

    # load file and voxelize
    savedir = os.path.join(args.savedir, "test", "pointcloud_txt")
    os.makedirs(savedir, exist_ok=True)
    Sem3D.semantic3d_load_from_txt_voxel(os.path.join(args.rootdir, "TEST", filename_txt),
                                                os.path.join(savedir, filename+"_voxels.txt"),
                                                args.voxel
                                                )
    
    # save the numpy data
    savedir_numpy = os.path.join(args.savedir, "test", "pointcloud")
    os.makedirs(savedir_numpy, exist_ok=True)
    np.save(os.path.join(savedir_numpy, filename+"_voxels"), np.loadtxt(os.path.join(savedir, filename+"_voxels.txt")).astype(np.float16))

print("Done")