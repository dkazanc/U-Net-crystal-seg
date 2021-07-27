from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import logging
from PIL import Image
import random


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, centercrop, scale, mask_suffix='gt'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mask_suffix = mask_suffix
        self.imgs_files = sorted(glob(imgs_dir + "*"))
        self.mask_files = sorted(glob(masks_dir + "*"))
        logging.info(f'Creating dataset with {len(self.imgs_files)} examples')
        self.transform = T.Compose([
            T.CenterCrop(centercrop)
        ])

    def __len__(self):
        return len(self.imgs_files)

    @classmethod
    def preprocess(cls, pil_img, scale, mode="img"):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1 and mode == "img":
            img_trans = (img_trans - img_trans.min()) / (img_trans.max() - img_trans.min())

        return img_trans

    def __getitem__(self, i):
        img_file = self.imgs_files[i]
        mask_file = self.mask_files[i]

        assert mask_file is not None, \
            f'The mask {mask_file} does not exist'
        assert img_file is not None, \
            f'The image {img_file} does not exist'
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image {img_file} and mask {mask_file} should be the same size, but are {img.size} and {mask.size}'

        # Transformations need to be done to PIL Images
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        if random.random() > 0.5:
            image = T.functional.hflip(img)
            mask = T.functional.hflip(mask)

        img = self.preprocess(img, self.scale, mode="img").astype(np.float32)
        mask = self.preprocess(mask, self.scale, mode="mask")

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        sample = {
            'image': img,
            'mask': mask
        }

        return sample
