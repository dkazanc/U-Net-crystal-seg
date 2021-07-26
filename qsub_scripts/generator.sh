echo "Loading conda environment"
#source ~/.bashrc_conda_dls_science
#conda activate /dls/science/users/kjy41806/miniconda/envs/pytorch
#export PYTHONPATH=/dls/science/users/kjy41806/miniconda/envs/pytorch/bin:$PYTHONPATH
module load savu/pre-release
cd /dls/science/users/kjy41806/crystal_segmentation/

input_folder=$1
output_folder=$2

echo "Begin generating data..."
python synth_generator.py -i $input_folder -m $output_folder -n 15 -s 900
echo "The job has been completed"
