import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(1)
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = net(img)
        
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
            probs = torch.argmax(probs, dim=1).float().cpu()
        else:
            probs = torch.sigmoid(output)

        if len(probs.shape) == 4:
            probs = probs.squeeze(0)
 
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='Folder of input images (/path/to/input/)', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', 
                        help='Folder of output images (/path/to/output/)')
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = glob.glob(args.input + "*")
    out_files = []
    
    for f in in_files:
        pathsplit = os.path.splitext(f)
        filename = pathsplit[0].split("/")[-1]
        out_files.append("{}_{}_OUT{}".format(args.output, filename, pathsplit[1]))

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = glob.glob(args.input + "*")
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=4)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
 
        img = Image.open(fn)
        
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)
        
        out_fn = out_files[i]
        result = mask_to_image(mask)
        result.save(out_files[i])

        logging.info("Mask saved to {}".format(out_files[i]))
