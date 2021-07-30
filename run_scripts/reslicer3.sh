#!/bin/bash
########### CHANGE PARAMETERS HERE ###########
path_to_datasets="../DATA_UNET_TOMO/"
output_path_to_resliced_data="../DATA_UNET_TOMO/"
name_of_the_dataset="resliced"
number_of_datasets=10 # specify how many datasets you want to reslice
save_style="one" # specify how the data needs to be saved, into "one" folder or "three" folders
#############################################

echo "Begin reslicing of training data"
python utils/3axes_generator.py -i $path_to_datasets -o $output_path_to_resliced_data -d $name_of_the_dataset -n $number_of_datasets -s $save_style
echo "Reslicing complete"
