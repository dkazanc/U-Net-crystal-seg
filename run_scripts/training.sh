#!/bin/bash
path_to_reconstructions=/home/algol/Documents/DEV/DATA_TEMP/UNET_DATA/128_data/3axis_XY_YZ_XZ/recon/
path_to_masks=/home/algol/Documents/DEV/DATA_TEMP/UNET_DATA/128_data/3axis_XY_YZ_XZ/gt/
path_to_checkpoints=/home/algol/Documents/DEV/DATA_TEMP/UNET_DATA/128_data/checkpoints/
number_of_epochs=30
batch_size=30 # find the largest batch_size which would fit your GPU card
learning_rate=1e-05
croping_size=128 # specifiy the cropping size of the data, use the original image size if no cropping required

echo "Begin training..."
python train.py -e $number_of_epochs -b $batch_size -l $learning_rate -i $path_to_reconstructions -m $path_to_masks -c $path_to_checkpoints -crop $croping_size
echo "Training has completed"
