import numpy as np
import semantic3D_utils.lib.python.semantic3D as sem3D
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testdir', '-s', help='Path to data folder')
parser.add_argument("--savedir", type=str, default="./results")
parser.add_argument("--refdata", type=str, default="./results")
parser.add_argument("--reflabel", type=str, default="./results")
args = parser.parse_args()

filenames = [
        ["birdfountain_station1_xyz_intensity_rgb","birdfountain1.labels"],
        ["castleblatten_station1_intensity_rgb","castleblatten1.labels"],
        ["castleblatten_station5_xyz_intensity_rgb","castleblatten5.labels"],
        ["marketplacefeldkirch_station1_intensity_rgb","marketsquarefeldkirch1.labels"],
        ["marketplacefeldkirch_station4_intensity_rgb","marketsquarefeldkirch4.labels"],
        ["marketplacefeldkirch_station7_intensity_rgb","marketsquarefeldkirch7.labels"],
        ["sg27_station10_intensity_rgb","sg27_10.labels"],
        ["sg27_station3_intensity_rgb","sg27_3.labels"],
        ["sg27_station6_intensity_rgb","sg27_6.labels"],
        ["sg27_station8_intensity_rgb","sg27_8.labels"],
        ["sg28_station2_intensity_rgb","sg28_2.labels"],
        ["sg28_station5_xyz_intensity_rgb","sg28_5.labels"],
        ["stgallencathedral_station1_intensity_rgb","stgallencathedral1.labels"],
        ["stgallencathedral_station3_intensity_rgb","stgallencathedral3.labels"],
        ["stgallencathedral_station6_intensity_rgb","stgallencathedral6.labels"],
]

os.makedirs(args.savedir, exist_ok=True)

for fname in filenames:
    print(fname[0])
    data_filename = os.path.join(args.testdir, fname[0]+".txt")
    dest_filaname = os.path.join(args.savedir, fname[1])
    refdata_filename = os.path.join(args.refdata, fname[0]+"_voxels.txt")
    reflabel_filename = os.path.join(args.reflabel, fname[0]+"_voxels.npy")

    sem3D.project_labels_to_pc(dest_filaname, data_filename, refdata_filename, reflabel_filename)