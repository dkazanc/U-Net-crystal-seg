import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from unet import UNet
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()

    full_img = Image.fromarray(full_img)
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(1)
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = net(img)
        
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
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
    in_files = sorted(glob.glob(args.input + "*"))
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
    in_files = sorted(glob.glob(args.input + "*"))
    out_files = get_output_filenames(args)
    
    print("Number of files:", len(in_files))

    net = UNet(n_channels=1, n_classes=4)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    
    # Load volume into memory
    print("Loading volume into memory...")
    sample = []
    for n in range(len(in_files)):
        img = np.array(Image.open(in_files[n]))
        sample.append(img)
    sample = np.array(sample)
    print("Done!")
    
    # XY prediction
    print("Predicting XY axis...")
    XY_volume = []
    for i in range(sample.shape[0]):
        mask = predict_img(net=net,
                           full_img=sample[i,:,:],
                           scale_factor=args.scale,
                           device=device)
        XY_volume.append(mask)
    XY_volume = np.array(XY_volume)
    print("Done!")
    
    # XZ prediction
    print("Predicting XZ axis...")
    XZ_volume = []
    for i in range(sample.shape[1]):
        mask = predict_img(net=net,
                           full_img=sample[:,i,:],
                           scale_factor=args.scale,
                           device=device)
        XZ_volume.append(mask)
    XZ_volume = np.array(XZ_volume)
    print("Done!")
    
    # YZ prediction
    print("Predicting YZ axis...")
    YZ_volume = []
    for i in range(sample.shape[2]):
        mask = predict_img(net=net,
                           full_img=sample[:,:,i],
                           scale_factor=args.scale,
                           device=device)
        YZ_volume.append(mask)
    YZ_volume = np.array(YZ_volume)
    print("Done!")
    
    XY_torch = torch.tensor(XY_volume).permute(1,0,2,3)
    XZ_torch = torch.tensor(XZ_volume).permute(1,2,0,3)
    YZ_torch = torch.tensor(YZ_volume).permute(1,2,3,0)
    
    predictions = np.stack([XY_torch, XZ_torch, YZ_torch])
    
    #predictions = np.max(predictions, axis=0) is worst than mean
    print("Getting average probability from different orentations...")
    predictions = np.mean(predictions, axis=0) #better results than max
    predictions = torch.argmax(torch.tensor(predictions), dim=0)
    predictions = np.array(predictions)
    print("Done!")
    
    print("SAVING...")
    for n in tqdm(range(predictions.shape[0])):
        filename = out_files[n]
        im = mask_to_image(predictions[n])
        im.save(filename)
    

