# distutils: language = c++
# distutils: sources = Semantic3D.cxx

import numpy as np
cimport numpy as np
import cython

cdef extern from "Sem3D.h":

    void project_labels_to_point_cloud(
	char* output_filename,
	char* filename_pc, 
	char* filename_pc_with_labels, 
	char* filename_labels)

    void sem3d_from_txt_voxelize(char* filename, char* destination_filename, float voxel_size)
    void sem3d_from_txt_voxelize_labels(char* filename, char* labels_filename, char* destination_filename, float voxel_size)

def project_labels_to_pc(
    output_filename,
    input_points_filename,
    reference_points_filename,
    reference_labels_filename):

    project_labels_to_point_cloud(output_filename.encode(),
                         input_points_filename.encode(),
                         reference_points_filename.encode(),
                         reference_labels_filename.encode())

def semantic3d_load_from_txt_voxel(filename, filename_dest, voxel_size):
    cdef bytes filename_bytes = filename.encode()
    cdef bytes filename_dest_bytes = filename_dest.encode()
    sem3d_from_txt_voxelize(filename_bytes, filename_dest_bytes, voxel_size)

def semantic3d_load_from_txt_voxel_labels(filename, filename_labels, filename_dest, voxel_size):
    cdef bytes filename_bytes = filename.encode()
    cdef bytes filename_labels_bytes = filename_labels.encode()
    cdef bytes filename_dest_bytes = filename_dest.encode()
    sem3d_from_txt_voxelize_labels(filename_bytes, filename_labels_bytes, filename_dest_bytes, voxel_size)