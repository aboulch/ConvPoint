# NPM3D example

## Data

Data can be downloaded at [http://npm3d.fr/paris-lille-3d](http://npm3d.fr/paris-lille-3d).

The script ```npm3d_prepare_data.py``` can be used to split the point clouds and save them at numpy format.

For the training set:
```
python npm3d_prepare_data.py --rootdir path_to_data --destdir path_to_data_processed
```

For the test set:
```
python npm3d_prepare_data.py --rootdir path_to_data --destdir path_to_data_processed --test
```

## Training
```
python npm3d_seg.py --rootdir path_to_data_dir --savedir path_to_save_dir
```
```
python npm3d_seg.py --rootdir path_to_data_dir --savedir path_to_save_dir --nocolor
```

## Testing

```
python npm3d_seg.py --rootdir path_to_data_dir --savedir path_to_save_dir --test
```

If the model was trained without using the lidar intensity:
```
python npm3d_seg.py --rootdir path_to_data_dir --savedir path_to_save_dir --test --nocolor
```

**note**: the `test_step` parameter is set `0.8`. It is possible to change it. A smaller step of sliding window would produce better segmentation at a the cost of a longer computation time.

## Fusion model

Once models (RGB and without color information) have been trained, it is possible to train a fusion model.

### Training
```
python npm33d_seg_fusion.py --rootdir path_to_data_processed --savedir path_to_save_dirctory --model_rgb path_to_rgb_model_directory --model_noc path_to_no_color_model_directory
```
### Test
```
python npm3d_seg_fusion.py --rootdir path_to_data_processeed --savedir path_to_save_dirctory --model_rgb path_to_rgb_model_directory --model_noc path_to_no_color_model_directory --test --savepts
```