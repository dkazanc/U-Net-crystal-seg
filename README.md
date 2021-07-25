# U-net segmentation of synthetic and real macromolecular crystallography tomographic data

## Description:
This is a Pytorch implementation of U-net semantic segmentation adapted from this [repository](https://github.com/milesial/Pytorch-UNet) to the case of segmenting macromolecular crystallography tomographic data.

### Contents:
* [Synthetic data generator](https://github.com/dkazanc/U-Net-crystal-seg/blob/main/synth_data_gen/synth_data_generator.py) script uses [Tomophantom](https://github.com/dkazanc/TomoPhantom) package to generate multiple 3D phantoms then adds realistic imaging artifacts to the projection data and apply [ToMoBAR](https://github.com/dkazanc/ToMoBAR) package to iteratively reconstruct. The resulting data for ground truth masks and the reconstructed images saved into image stacks. From the main folder run:
```
python synth_data_gen/synth_data_generator.py -i OUTPUT_PATH_TO_RECON -m OUTPUT_PATH_TO_MASKS -n NUMBER_of_DATASETS -s RECON_SIZE -a TOTAL_PROJECTIONS_NUMBER
```
* Script to generate 3 axes for training from the result of the synthetic data generator:
```
python utils/3axes_generator.py -i PATH_TO_DATASET -o OUTPUT_PATH -d NAME_OF_DATASET -n NUMBER_of_DATASETS
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
* [Gerard Jover Pujol](https://github.com/IararIV)
* [Daniil Kazantsev](https://github.com/dkazanc)

### License:
GNU GENERAL PUBLIC LICENSE v.3
