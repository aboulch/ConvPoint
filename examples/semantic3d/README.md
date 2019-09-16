# Semantic3D

## Data

Data can be downloaded at [http://semantic3d.net](http://semantic3d.net).

In the folder ```semantic3D_utils```:
```
python setup.py install --home="."
```
Then, run the generation script:
```
python semantic3d_prepare_data.py --rootdir path_to_data_dir --savedir path_to_data_processed
```
## Training

The training script is called using:
```
python semantic3d_seg.py --rootdir path_to_data_processed --savedir path_to_save_dir
```

## Test

To predict on the test set (voxelized pointcloud):

```
python semantic3d_seg.py --rootdir path_to_data_processed --savedir path_to_save_dir --test
```

Finally to generate the prediction files at benchmark format (may take som time): 

```
python semantic3d_benchmark_gen.py --testdir path_to_original_test_data --savedir /path_to_save_dir_benchmark --refdata path_to_data_processed --reflabel path_to_prediction_dir
```