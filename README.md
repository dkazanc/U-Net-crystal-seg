# Deep learning segmentation of synthetic and real tomographic data using Pytorch U-net

## Description:
This code is adapted to the case of tomographic data semantic multi-class segmentation using the [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) repository. The 3D tomographic data for training is simulated with distortions and noise and then directly or iteratively reconstructed. U-net is then used to train on the reconstructed data and the ground truth masks to produce a trained model. The generated test data then predicted (segmented) while using the model.

### Synthetic data generator
* [Synthetic data generator](https://github.com/dkazanc/U-Net-crystal-seg/blob/main/synth_data_gen/synth_data_generator.py) script uses [Tomophantom](https://github.com/dkazanc/TomoPhantom) package to generate multiple 3D phantoms then adds realistic imaging artifacts to the projection data and apply [ToMoBAR](https://github.com/dkazanc/ToMoBAR) package to iteratively reconstruct. The resulting data for ground truth masks and the reconstructed images saved into image stacks. From the main folder run:
```
python synth_data_gen/synth_data_generator.py -i OUTPUT_PATH_to_RECON -m OUTPUT_PATH_to_MASKS -n NUMBER_of_DATASETS -s RECON_SIZE -a TOTAL_PROJECTIONS_NUMBER
```
* Script to generate 3 axes for training from the result of the synthetic data generator:
```
python utils/3axes_generator.py -i PATH_to_DATASET -o OUTPUT_PATH -d NAME_of_DATASET -n NUMBER_of_DATASETS -s ONE_or_THREE_FOLDERS
```
### U-net training
* After synthetic data has been generated you can train the model:
```
python train.py -e NUMBER_of_EPOCHS -b BATCH_SIZE -l LEARNING_RATE -i PATH_to_DATASET -m PATH_to_MASKS -c PATH_to_CHECKPOINTS -cr CROPPING_SIZE
```
* One can use Tensorboard to check the loss and the learning rate
```
tensorboard --logdir=./ --bind_all
```

### Prediction

* After training one can use the resulting model (a checkpoint) in application to the test dataset:
```
python predict.py -i PATH_to_TESTDATA -o OUTPUT_PATH -m PATH_to_MODEL
```

### Installation (Linux):
* Git clone the repo: `git clone https://github.com/dkazanc/U-Net-crystal-seg.git`
* Install miniconda and use conda explicit file to install all required packages into the `unet_pytorch` environment:
```
conda create --name unet_pytorch --file conda_environment/spec_unet_pytorch.txt
```
* Activate the environment `conda activate unet_pytorch`

### Software dependencies:
 * [Pytorch](https://pytorch.org/), Tesorboard, Seaborn
 * [ASTRA-toolbox](https://www.astra-toolbox.com/) for forward/backward tomographic projection operations
 * [TomoPhantom](https://github.com/dkazanc/TomoPhantom) for tomographic data and phantoms simulations
 * [ToMoBAR](https://github.com/dkazanc/ToMoBAR) for iterative tomographic reconstruction
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) for regularisation


### Contributors:
* [Gerard Jover Pujol](https://github.com/IararIV) (as a part of Year in Industry project, 2020-21 at Diamond Light Source)
* [Daniil Kazantsev](https://github.com/dkazanc) (supervisor)

### License:
GNU GENERAL PUBLIC LICENSE v.3
