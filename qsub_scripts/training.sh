echo "Loading conda environment"
source ~/.bashrc_conda_dls_science
conda activate /dls/science/users/kjy41806/miniconda/envs/pytorch

export PYTHONPATH=/dls/science/users/kjy41806/miniconda/envs/pytorch/bin:$PYTHONPATH
cd /dls/science/users/kjy41806/crystal_segmentation/U_Net

data_recon_folder=$1
data_masks_folder=$2
checkpoints=$3
epochs_number=$4
batch_size=$5
learning_rate=$6
gpu_number=$7

echo "Begin Training..."
python train.py -e $epochs_number -b $batch_size -l $learning_rate -i $data_recon_folder -m $data_masks_folder -gpu $gpu_number -c $checkpoints
#nvidia-smi
#sleep 1
echo "The job has been completed"
