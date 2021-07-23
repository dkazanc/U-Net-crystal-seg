echo "Loading conda environment"
source ~/.bashrc_conda_dls_science
conda activate /dls/science/users/kjy41806/miniconda/envs/pytorch

export PYTHONPATH=/dls/science/users/kjy41806/miniconda/envs/pytorch/bin:$PYTHONPATH
cd /dls/science/users/kjy41806/crystal_segmentation/U_Net

model=$1
input_folder=$2
output_folder=$3

echo "Begin prediction..."
python predict.py -i $input_folder -o $output_folder -m $model

echo "The job has been completed"
