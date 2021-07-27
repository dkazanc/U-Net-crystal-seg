model=/home/algol/Documents/DEV/DATA_TEMP/UNET_DATA/128_data/checkpoints/CP_epoch30.pth
input_folder=/home/algol/Documents/DEV/DATA_TEMP/UNET_DATA/128_data/testdata/recon/
output_folder=/home/algol/Documents/DEV/DATA_TEMP/UNET_DATA/128_data/testdata/prediction/

echo "Begin prediction..."
python predict_volume_probs.py -i $input_folder -o $output_folder -m $model
echo "The job has been completed"
