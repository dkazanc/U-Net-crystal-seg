# run the code ./training_run.sh training.sh $checkpoints_path
module load hamilton 

script_to_run=$1
checkpoints=$2

data_recon_folder=/dls/i23/data/2021/cm28128-1/processing/tomography/recon/14254/daniil/TRAIN/reconXZ_YZ/
data_masks_folder=/dls/i23/data/2021/cm28128-1/processing/tomography/recon/14254/daniil/TRAIN/gtXZ_YZ/
# parameters for training
epochs_number=50
batch_size=6
learning_rate=1e-05
gpu_number=0

qsub -q all.q -P tomography -l gpu=1 -l gpu_arch=Volta $script_to_run $data_recon_folder $data_masks_folder $checkpoints $epochs_number $batch_size $learning_rate $gpu_number

