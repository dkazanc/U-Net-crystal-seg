import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# python test.py -p /dls/tmp/lqg38422/TEST/gt/ -m /dls/tmp/lqg38422/TEST/gt/

def get_args():
    parser = argparse.ArgumentParser(description='Get metrics to evaluate the predictions of the U-Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--dir_pred', dest='dir_pred', type=str, default='/dls/tmp/lqg38422/PREDS/',
                        help='Path to the folder containing the images (/path/to/preds/)')
    parser.add_argument('-m', '--dir_mask', dest='dir_mask', type=str, default='/dls/tmp/lqg38422/TEST/gt/',
                        help='Path to the folder containing the masks (/path/to/masks/)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    preds_files = glob.glob(args.dir_pred + "*")
    masks_files = glob.glob(args.dir_mask + "*")
    
    pred_tensor = []
    mask_tensor = []
    
    for n in range(len(preds_files)):
        pred = np.array(Image.open(preds_files[n]))
        mask = np.array(Image.open(masks_files[n]))
        pred_tensor.append(pred)
        mask_tensor.append(mask)
        
    pred_tensor = np.array(pred_tensor)
    mask_tensor = np.array(mask_tensor)
    
    confusion_matrix = np.zeros((4,4))
    
    # pred is 0 and mask is 0 / mask is 0
    # pred is 1 and mask is 0
    
    # True background
    
    confusion_matrix[0,0] = np.sum((pred_tensor == 0) & (mask_tensor == 0)) / np.sum(mask_tensor == 0)
    confusion_matrix[0,1] = np.sum((pred_tensor == 1) & (mask_tensor == 0)) / np.sum(mask_tensor == 0)
    confusion_matrix[0,2] = np.sum((pred_tensor == 2) & (mask_tensor == 0)) / np.sum(mask_tensor == 0)
    confusion_matrix[0,3] = np.sum((pred_tensor == 3) & (mask_tensor == 0)) / np.sum(mask_tensor == 0)
    
    # True crystal
    confusion_matrix[1,0] = np.sum((pred_tensor == 0) & (mask_tensor == 1)) / np.sum(mask_tensor == 1)
    confusion_matrix[1,1] = np.sum((pred_tensor == 1) & (mask_tensor == 1)) / np.sum(mask_tensor == 1)
    confusion_matrix[1,2] = np.sum((pred_tensor == 2) & (mask_tensor == 1)) / np.sum(mask_tensor == 1)
    confusion_matrix[1,3] = np.sum((pred_tensor == 3) & (mask_tensor == 1)) / np.sum(mask_tensor == 1)
    
    # True loop
    confusion_matrix[2,0] = np.sum((pred_tensor == 0) & (mask_tensor == 2)) / np.sum(mask_tensor == 2)
    confusion_matrix[2,1] = np.sum((pred_tensor == 1) & (mask_tensor == 2)) / np.sum(mask_tensor == 2)
    confusion_matrix[2,2] = np.sum((pred_tensor == 2) & (mask_tensor == 2)) / np.sum(mask_tensor == 2)
    confusion_matrix[2,3] = np.sum((pred_tensor == 3) & (mask_tensor == 2)) / np.sum(mask_tensor == 2)
    
    # True liquor
    confusion_matrix[3,0] = np.sum((pred_tensor == 0) & (mask_tensor == 3)) / np.sum(mask_tensor == 3)
    confusion_matrix[3,1] = np.sum((pred_tensor == 1) & (mask_tensor == 3)) / np.sum(mask_tensor == 3)
    confusion_matrix[3,2] = np.sum((pred_tensor == 2) & (mask_tensor == 3)) / np.sum(mask_tensor == 3)
    confusion_matrix[3,3] = np.sum((pred_tensor == 3) & (mask_tensor == 3)) / np.sum(mask_tensor == 3)
        
    df_cm = pd.DataFrame(confusion_matrix, index = ["Background", "Crystal", "Loop", "Liquor"],
                  columns = ["Background", "Crystal", "Loop", "Liquor"])
    
    plt.figure(figsize = (10,7))
    plt.title('Confusion matrix')
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
