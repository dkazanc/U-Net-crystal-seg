import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import glob
import shutil
import time

"""
Script to generate 3 sets of image stacks for different slicing, the input data should have
"recon" and "gt" folders
Usage >>>> python 3axes_generator.py -i PATH_TO_DATASET -o OUTPUT_PATH -d NAME_OF_DATASET
Authors:
Gerard Jover Pujol
Daniil Kazantsev
"""


def get_args():
    parser = argparse.ArgumentParser(description='Create the folder with the training data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--dir_data', dest='dir_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('-o', '--dir_output', dest='dir_output', type=str,
                        help='Path to the folder where images will be stored')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        help='Name of the dataset')
    parser.add_argument('-s', '--save_method', dest='save_method', type=str, default='three',
                        help='How to save the output data, "one" folder or "three" separate')

    return parser.parse_args()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    logging.info(f'Data:\n'
                 f'\tDataset directory : {args.dir_data}\n'
                 f'\tImage output directory : {args.dir_output}\n'
                 f'\tDataset : {args.dataset}\n')

    dataset = args.dataset
    output_path = args.dir_output
    save_method = args.save_method

    # Create output folders
    if (save_method == 'three'):
        # create 3 separate folders
        Path(output_path + dataset + "_XY/recon/").mkdir(parents=True, exist_ok=True)
        Path(output_path + dataset + "_XY/gt/").mkdir(parents=True, exist_ok=True)

        Path(output_path + dataset + "_XZ/recon/").mkdir(parents=True, exist_ok=True)
        Path(output_path + dataset + "_XZ/gt/").mkdir(parents=True, exist_ok=True)

        Path(output_path + dataset + "_YZ/recon/").mkdir(parents=True, exist_ok=True)
        Path(output_path + dataset + "_YZ/gt/").mkdir(parents=True, exist_ok=True)
    else:
        # create 1 folder only (3 axes in one desitnation)
        Path(output_path + dataset + "_XY_YZ_XZ/recon/").mkdir(parents=True, exist_ok=True)
        Path(output_path + dataset + "_XY_YZ_XZ/gt/").mkdir(parents=True, exist_ok=True)

    # Add training data
    t = time.time()
    print(f"--- DATASET {dataset} ---")

    recon_path = args.dir_data + "/recon/"
    gt_path = args.dir_data + "/gt/"

    # Load the tensors (reconstruction and ground truth) into memory
    print("Loading tensors into memory...")
    sample = []
    segment = []
    for filename in sorted(glob.glob(recon_path + "*")):
        sample.append(np.array(Image.open(filename)))
    sample = np.array(sample)
    if sample.dtype == np.float32:
        print("Please normalise the imput (recon) data and save it to uint16 first")
        exit()
    for filename in sorted(glob.glob(gt_path + "*")):
        im = np.array(Image.open(filename))
        segment.append(im)
    segment = np.array(segment)
    print("Done!")
    # Save for each axis
    print("XY axis:")
    for xy in range(len(sample[:,0,0])):
        # Recon
        filename = dataset + "XY_recon_" + str(xy).zfill(5) + ".tif"
        im = Image.fromarray(sample[xy,:,:].astype(np.uint16))
        if (save_method == 'three'):
            im.save(output_path + dataset + "_XY/recon/" + filename)
        else:
            im.save(output_path + dataset + "_XY_YZ_XZ/recon/" + filename)
        # Ground truth
        filename = dataset + "XY_gt_" + str(xy).zfill(5) + ".tif"
        im = Image.fromarray(segment[xy,:,:].astype(np.uint8))
        if (save_method == 'three'):
            im.save(output_path + dataset + "_XY/gt/" + filename)
        else:
            im.save(output_path + dataset + "_XY_YZ_XZ/gt/" + filename)
    print("Done!")
    print("XZ axis:")
    for xz in range(len(sample[0,:,0])):
        # Recon
        filename = dataset + "XZ_recon_" + str(xz).zfill(5) + ".tif"
        im = Image.fromarray(sample[:,xz,:].astype(np.uint16))
        if (save_method == 'three'):
            im.save(output_path + dataset + "_XZ/recon/" + filename)
        else:
            im.save(output_path + dataset + "_XY_YZ_XZ/recon/" + filename)
        # Ground truth
        filename = dataset + "XZ_gt_" + str(xz).zfill(5) + ".tif"
        im = Image.fromarray(segment[:,xz,:].astype(np.uint8))
        if (save_method == 'three'):
            im.save(output_path + dataset + "_XZ/gt/" + filename)
        else:
            im.save(output_path + dataset + "_XY_YZ_XZ/gt/" + filename)
    print("Done!")
    print("YZ axis:")
    for yz in range(len(sample[0,0,:])):
        # Recon
        filename = dataset + "YZ_recon_" + str(yz).zfill(5) + ".tif"
        im = Image.fromarray(sample[:,:,yz].astype(np.uint16))
        if (save_method == 'three'):
            im.save(output_path + dataset + "_YZ/recon/" + filename)
        else:
            im.save(output_path + dataset + "_XY_YZ_XZ/recon/" + filename)
        # Ground truth
        filename = dataset + "YZ_gt_" + str(yz).zfill(5) + ".tif"
        im = Image.fromarray(segment[:,:,yz].astype(np.uint8))
        if (save_method == 'three'):
            im.save(output_path + dataset + "_YZ/gt/" + filename)
        else:
            im.save(output_path + dataset + "_XY_YZ_XZ/gt/" + filename)
    print("Done!")

    res = time.time() - t
    print("Time:", res)
