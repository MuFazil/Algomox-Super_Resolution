# Algomox-Super_Resolution

## Dataset folder contains the dataset used for both training and testing

## SRGAN_HPT
#### Is for Hyper Parameter Tuning
command : nnictl create --config path\to\config.yaml
#### Only a sample dataset is used for this task
#### Parameter range and choices can be adjusted in the search space


## SRGAN_NAS
#### Is for Neural Architecture Search
command : nnictl create --config path\to\config.yaml
#### Only a sample dataset is used for this task
#### Parameter range and choices can be adjusted in the search space


## SRGAN_Training
#### Contains files for training (both pre-training and finetuning),  validation and testing
Training command : !python main.py --LR_path SRGAN_CustomDataset\custom_dataset\train_LR --GT_path SRGAN_CustomDataset\custom_dataset\train_HR

Testing Command : !python main.py main.py --mode test --LR_path Dataset\valid\LowRes --GT_path Dataset\valid\HighRes --generator_path SRGAN_CustomDataset\model\pre_trained_model_2700.pt
#### Check the available command line arguments in main.py


## Python Scripts for Data Pre-processing
#### Augment_RC.py
#### downsample.py
#### GrayScaleRM.py
