#!/bin/bash
mkdir ../DATA_UNET_TOMO/TESTDATA/prediction
rm -rf ../DATA_UNET_TOMO/TESTDATA/prediction/*
########### CHANGE PARAMETERS HERE ###########
model="../DATA_UNET_TOMO/resliced_XY_YZ_XZ/checkpoints/CP_epoch30.pth"
input_folder="../DATA_UNET_TOMO/TESTDATA/recon/"
output_folder="../DATA_UNET_TOMO/TESTDATA/prediction/"

#############################################
echo "Begin prediction..."
python predict_volume_probs.py -i $input_folder -o $output_folder -m $model
echo "The job has been completed"
