# ConvPoint: Generalizing discrete convolutions for unstructured point clouds


![SnapNet products](./doc/convPoint.png)


## Updates

**Major performance update**: by reformulating the convolutional layer using matrix mulitplications, the memory consumption has been highly reduced.

**Major interface update**: the spatial relations are now computed in the network class. The framework is then easier to use and more flexible.

## Introduction

This repository propose python scripts for point cloud classification and segmentation. The library is coded with PyTorch.

The journal paper is here:
http://www.sciencedirect.com/science/article/pii/S0097849320300224

The conference paper is here:
https://diglib.eg.org/handle/10.2312/3dor20191064

A preprint of the paper can be found on Arxiv:  
http://arxiv.org/abs/1904.02375

## License

Code is released under dual license depending on applications, research or commercial. Reseach license is GPLv3.
See the [license](LICENSE.md).

## Citation

If you use this code in your research, please consider citing:

```
@article{BOULCH202024,
title = "ConvPoint: Continuous convolutions for point cloud processing",
journal = "Computers & Graphics",
volume = "88",
pages = "24 - 34",
year = "2020",
issn = "0097-8493",
doi = "https://doi.org/10.1016/j.cag.2020.02.005",
url = "http://www.sciencedirect.com/science/article/pii/S0097849320300224",
author = "Alexandre Boulch",
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
