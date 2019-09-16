#ifndef _Sem3D_h
#define _Sem3D_h

#include <string>
#include <vector>

void project_labels_to_point_cloud(
	char* output_filename,
	char* filename_pc, 
	char* filename_pc_with_labels, 
	char* filename_labels);

void sem3d_from_txt_voxelize(char* filename, char* destination_filename, float voxel_size);

void sem3d_from_txt_voxelize_labels(char* filename, char* labels_filename, char* destination_filename, float voxel_size);

#endif
