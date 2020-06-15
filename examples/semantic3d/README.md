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

**note**: the `test_step` parameter is set `0.8`. It is possible to change it. A smaller step of sliding window would produce better segmentation at a the cost of a longer computation time.

## Fusion model

Once models (RGB and without color information) have been trained, it is possible to train a fusion model.

### Training
```
python semantic3d_seg_fusion.py --rootdir path_to_data_processed --savedir path_to_save_dirctory --model_rgb path_to_rgb_model_directory --model_noc path_to_no_color_model_directory
```
### Test
```
python semantic3d_seg_fusion.py --rootdir path_to_data_processeed --savedir path_to_save_dirctory --model_rgb path_to_rgb_model_directory --model_noc path_to_no_color_model_directory --test --savepts
```

## Pretrained models

Pretrained models can be found [here](https://github.com/aboulch/ConvPoint/releases/download/0.1.0/models_SEMANTIC3D_v0.zip).

**Note**: due to change of affiliation and loss of data, these models are given as they are, without any performance guarantee. Particularly, they may not be the ones used in the final version of the paper.