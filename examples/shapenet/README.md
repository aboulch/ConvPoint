# Shapenet part-segmentation example

## Getting the data


The script for downloading and preparing the point clouds are from *PointCNN* repository https://github.com/yangyanli/PointCNN[](https://github.com/yangyanli/PointCNN).

```
python3 ./download_datasets.py -d shapenet_partseg -f path_to_directory
python3 ./prepare_partseg_data.py -f path_to_shapenet_partseg

```

## Training

```
python shapenet_seg.py --rootdir path_to_data_dir --savedir path_to_save_dir --npoints 2500
```

## Testing

```
python shapenet_seg.py --rootdir path_to_data_dir --savedir path_to_save_dir --npoints 2500 --test
```

You can also use the ```--ply``` flag to generate PLY file for result visualization.

The previous command line is result with one spatial tree. To test with multiple spatial trees, use the  ```--ntree``` flag:

```
python shapenet_seg.py --rootdir path_to_data_dir --savedir path_to_save_dir --npoints 2500 --test --ntree 4
```