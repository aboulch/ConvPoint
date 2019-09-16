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