# ConvPoint: Generalizing discrete convolutions for unstructured point clouds

## Introduction

## License

Code is released under dual license depending on applications, research or commercial. Reseach license is GPLv3.

## Citation

If you use this code in your research, please consider citing:
(citation will be updated as soon as 3DOR proceedings will be released)

```
@inproceedings{boulch2019,
  title={Generalizing discrete convolutions for unstructured point clouds},
  author={Boulch, Alexandre},
  booktitle={Eurographics Workshop on 3D Object Retrieval},
  year={2019}
}
```

## Dependencies

- Pytorch
- Scikit-learn for confusion matrix computation, and efficient neighbors search
- Trimesh (for Modelnet40) for loading triangular meshes and sampling points    
- TQDM for progress bars

All these dependencies can be install via conda in an Anaconda environment or via pip.

## The library

## Usage

We propose scripts for training on several point cloud datasets:
- ModelNet40 (meshes can be found [here](http://modelnet.cs.princeton.edu/)). The meshes are sampled in the code using Trimesh.
- ShapeNet *(code to be added)*
- S3DIS *(code to be added)*
- Semantic8 *(code to added)*

### ModelNet40

#### Training
```
python modelnet_classif.py --rootdir path_to_modelnet40_data
```

#### Testing

For testing with one tree per shape:
```
python modelnet_classif.py --rootdir path_to_modelnet40_data --savedir path_to_statedict_directory --test
```
For testing with more than one tree per shape: *(this code is not optimized at all and is very slow)*
```
python modelnet_classif.py --rootdir path_to_modelnet40_data --savedir path_to_statedict_directory --test --ntree 2
```

