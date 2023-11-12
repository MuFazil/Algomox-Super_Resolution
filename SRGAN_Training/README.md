# Training Folder

## All parameters are adjusted accroding to NAS and HPT

### Training :

!python main.py --LR_path SRGAN_CustomDataset\custom_dataset\train_LR --GT_path SRGAN_CustomDataset\custom_dataset\train_HR

### Testing :

!python main.py main.py --mode test --LR_path Dataset\valid\LowRes --GT_path Dataset\valid\HighRes --generator_path SRGAN_CustomDataset\model\pre_trained_model_2700.pt
