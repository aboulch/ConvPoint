# ConvPoint: Generalizing discrete convolutions for unstructured point clouds


![SnapNet products](./doc/convPoint.png)


## Updates

**Major performance update**: by reformulating the convolutional layer using matrix mulitplications, the memory consumption has been highly reduced.

**Major interface update**: the spatial relations are now computed in the network class. The framework is then easier to use and more flexible.

## Introduction

This repository propose python scripts for point cloud classification and segmentation. The library is coded with PyTorch.

The conference paper is here:
https://diglib.eg.org/handle/10.2312/3dor20191064

A preprint of the paper can be found on Arxiv:  
http://arxiv.org/abs/1904.02375

## License

Code is released under dual license depending on applications, research or commercial. Reseach license is GPLv3.
See the [license](LICENSE.md).

## Citation

If you use this code in your research, please consider citing:
(citation will be updated as soon as 3DOR proceedings will be released)

```
@inproceedings {or.20191064,
booktitle = {Eurographics Workshop on 3D Object Retrieval},
editor = {Biasotti, Silvia and Lavoué, Guillaume and Veltkamp, Remco},
title = {{Generalizing Discrete Convolutions for Unstructured Point Clouds}},
author = {Boulch, Alexandre},
year = {2019},
publisher = {The Eurographics Association},
ISSN = {1997-0471},
ISBN = {978-3-03868-077-2},
DOI = {10.2312/3dor.20191064}
}
```

## Platform

The code was tested on Ubuntu 16.04 with Anaconda.

## Dependencies

- Pytorch
- Scikit-learn for confusion matrix computation, and efficient neighbors search  
- TQDM for progress bars
- PlyFile
- H5py

All these dependencies can be install via conda in an Anaconda environment or via pip.

## The library

### Nearest neighbor module

The ```nearest_neighbors``` directory contains a very small wrapper for [NanoFLANN](https://github.com/jlblancoc/nanoflann) with OpenMP.
To compile the module:
```
cd nearest_neighbors
python setup.py install --home="."
```

In the case, you do not want to use this C++/Python wrapper. You still can use the previous version of the nearest neighbors computation with Scikit Learn and Multiprocessing, python only version (slower). To do so, add the following lines at the start of your main script (e.g. ```modelnet_classif.py```):
```
from global_tags import GlobalTags
GlobalTags.legacy_layer_base(True)
```

## Examples
* [ModelNet40](examples/modelnet/)
* [ShapeNet](examples/shapenet/)
* [S3DIS](examples/s3dis/)
* [Semantic3D](examples/semantic3d)
* [NPM3D](examples/npm3d)
