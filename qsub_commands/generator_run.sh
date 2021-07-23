# run the code ./training_run.sh training.sh $checkpoints_path
module load hamilton 

data_recon_folder=/dls/i23/data/2021/cm28128-1/processing/tomography/recon/14254/daniil/TRAIN_synth/recon/
data_masks_folder=/dls/i23/data/2021/cm28128-1/processing/tomography/recon/14254/daniil/TRAIN_synth/gt/

qsub -q all.q -P tomography -l gpu=1 -l gpu_arch=Volta generator.sh $data_recon_folder $data_masks_folder

