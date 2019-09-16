# ModelNet40 example


[Modelnet40](https://modelnet.cs.princeton.edu/) is a dataset from Princeton Unviversity for point cloud classification.

Please download the HDF5 file at [https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)

## Training

```
python modelnet_classif.py --rootdir path_to_data --savedir path_to_save_directory --npoints 2048
```

## Testing

```
python modelnet_classif.py --rootdir path_to_data --savedir path_to_save_directory --npoints 2048 --test
```

For testing with more than one tree per shape: *(this code is not optimized at all and is very slow)*

```
python modelnet_classif.py --rootdir path_to_data --savedir path_to_save_directory --npoints 2048 --test --ntree 2
```

