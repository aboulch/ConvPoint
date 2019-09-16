# S3DIS Example


## Data preparation

Data is prepared using the ```prepare_s3dis_label.py```.

## Single model

### Training

For training on area 2:

```
python s3dis_seg.py --rootdir path_to_data_processed/ --area 2 --savedir path_to_save_directory
```

To train without color information (used for fusion):
```
python s3dis_seg.py --rootdir path_to_data_processed/ --area 2 --savedir path_to_save_directory --nocolor
```

### Test

For testing on area 2:
```
python s3dis_seg.py --rootdir path_to_data_processed --area 2 --savedir path_to_save_directory --test
```
If the ```--nocolor``` option was used at training, it should be also used during test:
```
python s3dis_seg.py --rootdir path_to_data_processed --area 2 --savedir path_to_save_directory --nocolor --test
```

## Fusion

### Training
After training of both color model and segmentation model, you can train a fusion model with:
```
python s3dis_seg_fusion.py --rootdir path_to_data_processed --area 2 --savedir path_to_save_directory --model_rgb path_to_model_rgb_dir --model_noc path_to_model_nocolor_dir
```
### Test
```
python s3dis_seg_fusion.py --rootdir path_to_data_processed --area 2 --savedir path_to_fusion_model_dir --model_rgb path_to_model_rgb_dir --model_noc path_to_model_nocolor_dir --test
```

## Evaluation

```
python s3dis_eval.py --datafolder path_to_data_processed --predfolder pathèto_model --area 2
```