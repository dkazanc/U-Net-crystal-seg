# run the code
module load hamilton 

model_path=$1
input_f=/dls/i23/data/2021/cm28128-1/processing/tomography/recon/14254/daniil/TEST/13284_XY/recon/
output_f=/dls/science/users/kjy41806/crystal_segmentation/U_Net/prediction/13284_XY/3axis/

qsub -q all.q -P tomography -l gpu=1 -l gpu_arch=Volta prediction.sh $model_path $input_f $output_f

