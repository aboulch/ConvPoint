#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

typedef std::vector<std::vector<double> > my_vector_of_vectors_t;



#include "Sem3D.h"
#include "Eigen/Dense"
#include <map>
#include <time.h>

using namespace std;

// class for voxels
class Voxel_center{
public:
    float x,y,z,d;
    int r,g,b;
    int intensity;
    int label;
};

// comparator for voxels
struct Vector3icomp {
	bool operator() (const Eigen::Vector3i& v1, const Eigen::Vector3i& v2) const{
		if(v1[0] < v2[0]){
			return true;
		}else if(v1[0] == v2[0]){
			if(v1[1] < v2[1]){
				return true;
			}else if(v1[1] == v2[1] && v1[2] < v2[2]){
				return true;
			}
		}
		return false;
	}
};

void sem3d_from_txt_voxelize(char* filename, char* destination_filename, float voxel_size){

    // open the semantic 3D file
    std::ifstream ifs(filename);
    std::string line;
    int pt_id =0;

    std::map<Eigen::Vector3i, Voxel_center, Vector3icomp> voxels;
    while(getline(ifs,line)){
        pt_id++;

        if(pt_id%100000==0){
            cout << "\r";
            cout << pt_id /1000000. << "M points loaded"<< std::flush;
        }

        std::stringstream sstr(line);
        float x,y,z;
        float intensity;
        int r, g, b;
        sstr >> x >> y >> z >> intensity >> r >> g >> b;

        int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y/voxel_size) + 0.5;
        int z_id = std::floor(z/voxel_size) + 0.5;

        Eigen::Vector3i vox(x_id, y_id, z_id);
        double d = (x-x_id)*(x-x_id) + (y-y_id)*(y-y_id) + (z-z_id)*(z-z_id);

        if(voxels.count(vox)>0){
            const Voxel_center& vc_ = voxels[vox];
            if(vc_.d > d){
                Voxel_center vc;
                vc.x = std::floor(x/voxel_size)*voxel_size;
                vc.y = std::floor(y/voxel_size)*voxel_size;
                vc.z = std::floor(z/voxel_size)*voxel_size;
                vc.d = d;
                vc.r = r;
                vc.g = g;
                vc.b = b;
                vc.intensity = intensity;
                voxels[vox] = vc;
            }

        }else{
            Voxel_center vc;
            vc.x = std::floor(x/voxel_size)*voxel_size;
            vc.y = std::floor(y/voxel_size)*voxel_size;
            vc.z = std::floor(z/voxel_size)*voxel_size;
            vc.d = d;
            vc.r = r;
            vc.g = g;
            vc.b = b;
            vc.intensity = intensity;
            voxels[vox] = vc;
        }
    }
    ifs.close();
    cout << endl;

    ofstream ofs (destination_filename);
    for(std::map<Eigen::Vector3i, Voxel_center>::iterator it=voxels.begin(); it != voxels.end(); it++){
        ofs << it->second.x << " ";
        ofs << it->second.y << " ";
        ofs << it->second.z << " ";
        ofs << it->second.r << " ";
        ofs << it->second.g << " ";
        ofs << it->second.b << " ";
        ofs << "0" << endl; // for the label
    }
    ofs.close();
}

void sem3d_from_txt_voxelize_labels(char* filename, char* labels_filename,
    char* destination_filename, float voxel_size){

    std::ifstream ifs(filename);
	std::ifstream ifs_labels(labels_filename);
	std::string line;
	std::string line_labels;
	int pt_id =0;

	std::map<Eigen::Vector3i, Voxel_center, Vector3icomp> voxels;
	while(getline(ifs,line)){
		pt_id++;
		getline(ifs_labels, line_labels);


        if(pt_id%100000==0){
            cout << "\r";
            cout << pt_id /1000000.  << "M points loaded"<< std::flush;
        }

		std::stringstream sstr_label(line_labels);
		int label;
		sstr_label >> label;

		// continue if points is unlabeled
		if(label == 0)
			continue;


		std::stringstream sstr(line);
		float x,y,z;
		int intensity;
		int r, g, b;
		sstr >> x >> y >> z >> intensity >> r >> g >> b;

		int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
		int y_id = std::floor(y/voxel_size) + 0.5;
		int z_id = std::floor(z/voxel_size) + 0.5;

		Eigen::Vector3i vox(x_id, y_id, z_id);
		double d = (x-x_id)*(x-x_id) + (y-y_id)*(y-y_id) + (z-z_id)*(z-z_id);

		if(voxels.count(vox)>0){
			const Voxel_center& vc_ = voxels[vox];
			if(vc_.d > d){
				Voxel_center vc;
				vc.x = std::floor(x/voxel_size)*voxel_size;
				vc.y = std::floor(y/voxel_size)*voxel_size;
				vc.z = std::floor(z/voxel_size)*voxel_size;
				vc.d = d;
				vc.r = r;
				vc.g = g;
				vc.b = b;
				vc.intensity = intensity;
				vc.label = label;
				voxels[vox] = vc;
			}

		}else{
			Voxel_center vc;
			vc.x = std::floor(x/voxel_size)*voxel_size;
			vc.y = std::floor(y/voxel_size)*voxel_size;
			vc.z = std::floor(z/voxel_size)*voxel_size;
			vc.d = d;
			vc.r = r;
			vc.g = g;
			vc.b = b;
			vc.intensity = intensity;
			vc.label = label;
			voxels[vox] = vc;
		}
	}
    ifs.close();
    ifs_labels.close();
    cout << endl;
    ofstream ofs (destination_filename);
    for(std::map<Eigen::Vector3i, Voxel_center>::iterator it=voxels.begin(); it != voxels.end(); it++){
        ofs << it->second.x << " ";
        ofs << it->second.y << " ";
        ofs << it->second.z << " ";
        ofs << it->second.r << " ";
        ofs << it->second.g << " ";
        ofs << it->second.b << " ";
        ofs << it->second.label << endl; // for the label
    }
    ofs.close();
}

void project_labels_to_point_cloud(
	char* output_filename,
	char* filename_pc, 
	char* filename_pc_with_labels, 
	char* filename_labels)
{
	// open the file stream
	ifstream ifs_labels(filename_labels);
	ifstream ifs_pts(filename_pc_with_labels);

	// load the points and labels
	my_vector_of_vectors_t  samples;
	std::vector<int> labels;
	string line;
	string line_label;
	cout << "Getting labeled points..." << endl;
	while(std::getline(ifs_pts, line)){

		// get the coordinates
		std::istringstream iss(line);
        float x,y,z;
        iss >> x >> y >> z;

		// get the label
		std::getline(ifs_labels, line_label);
		std::istringstream iss_label(line_label);
		int label;
		iss_label >> label;

		vector<double> point(3);
		point[0] = x;
		point[1] = y;
		point[2] = z;

		samples.push_back(point);
		labels.push_back(label);
	}
	ifs_labels.close();
	ifs_pts.close();
	cout << "Done " << samples.size() << " points" << endl;

	cout << "Create KDTree..." << endl;
	size_t dim=3;
	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t   mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
	mat_index.index->buildIndex();
	cout << "Done" << endl;

	cout << "Iteration on the original point cloud..." << endl; 
	ifstream ifs(filename_pc);
	ofstream ofs(output_filename);
    int pt_id = 0;
    while (std::getline(ifs, line))
    {   
		if((pt_id+1)%1000000==0){
			cout << "\r                              \r";
			cout << (pt_id+1)/1000000 << " M";
		}

		// get the query point coordinates
		std::istringstream iss(line);
        float x,y,z;
        iss >> x >> y >> z;
		std::vector<double> query_pt(3);
		query_pt[0] = x;
		query_pt[1] = y;
		query_pt[2] = z;

		// search for nearest neighbor
		const size_t num_results = 1;
		std::vector<size_t>   ret_indexes(num_results);
		std::vector<double> out_dists_sqr(num_results);
		nanoflann::KNNResultSet<double> resultSet(num_results);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		// get the label
		int label = labels[ret_indexes[0]];

        // write label in file // label 0 is unknow need to add 1
        ofs << label+1 << endl;

        // iterate point id
        pt_id ++;
    }

    // close input and output files
    ifs.close();
    ofs.close();
}

