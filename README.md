# U-net segmentation of macromolecular crystallography tomographic data

## Description:
This is a Pytorch implementation of U-net semantic segmentation adapted from this [repository](https://github.com/milesial/Pytorch-UNet) to the case of segmenting macromolecular crystallography tomographic data.

### Contents:
* [Synthetic data generator](https://github.com/dkazanc/U-Net-crystal-seg/blob/main/synth_data_gen/synth_data_generator.py) script which uses [Tomophantom](https://github.com/dkazanc/TomoPhantom) package to generate multiple 3D phantoms, adds realistic imaging artifacts to the projection data and apply [ToMoBAR](https://github.com/dkazanc/ToMoBAR) package to iteratively reconstruct projection data. The resulting data for ground truth masks and the reconstructed images saved into image stacks. Usage: `python synth_data_generator.py -i OUTPUT_PATH_TO_RECON -m OUTPUT_PATH_TO_MASKS -n NUMBER_of_DATASETS -s RECON_SIZE -a TOTAL_PROJECTIONS_NUMBER`


### Installation (Linux):
* Git clone the repo: `git clone https://github.com/dkazanc/U-Net-crystal-seg.git`
* Install miniconda and use conda explicit file to install all required packages into the `unet_pytorch` environment: `conda create --name unet_pytorch --file conda_environment/spec_unet_pytorch.txt`
* Activate the environment `conda activate unet_pytorch`

### Software dependencies:
 * [Pytorch](https://pytorch.org/), Tesorboard, Seaborn
 * [ASTRA-toolbox](https://www.astra-toolbox.com/) for forward/backward tomographic projection operations
 * [TomoPhantom](https://github.com/dkazanc/TomoPhantom) for tomographic data and phantoms simulations
 * [ToMoBAR](https://github.com/dkazanc/ToMoBAR) for iterative tomographic reconstruction
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) for regularisation

### License:
GNU GENERAL PUBLIC LICENSE v.3
