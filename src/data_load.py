# Reference: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


######################################################################
# Chicago Face Database
# ====================

class ChicagoFaceDatabase(Dataset):
    """Chicago Face Database."""

    def __init__(self, root_dir, transform=None, train=True):

        self.root_dir = root_dir
        self.transform = transform

        if train:
            self.landmarks_frame = pd.read_csv(os.path.join(self.root_dir, 'CFDbook-train.csv'))
        else:
            self.landmarks_frame = pd.read_csv(os.path.join(self.root_dir, 'CFDbook-test.csv'))

        self.image_dir = os.path.join(self.root_dir, 'Images/')

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        target = self.landmarks_frame.iloc[idx, 0]
        target_dir = os.path.join(self.image_dir, target)
        img_name = glob.glob(os.path.join(target_dir, '*.jpg'))
        image = io.imread(img_name[0])
        attractive = self.landmarks_frame.iloc[idx, 15].astype('double')

        attractive = int((attractive)/7*10-1)
        # CFD: 1-7 Likert, 1 = Not at all; 7 = Extremely
        # normalized attractive will be integer 0-9 (10 categories)

        if self.transform:
            image = self.transform(image)

        return (image, attractive)


######################################################################
# Instagram Selfie Database
# ====================

class InstagramDatabase(Dataset):
    """Instagram Selfie Database."""

    def __init__(self, root_dir, transform=None, train=True):

        self.root_dir = root_dir
        self.transform = transform

        if train:
            self.landmarks_frame = pd.read_csv(os.path.join(self.root_dir, 'image_info-train.csv'))
        else:
            self.landmarks_frame = pd.read_csv(os.path.join(self.root_dir, 'image_info-test.csv'))

        self.image_dir = os.path.join(self.root_dir, 'selfie/')

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        target = self.landmarks_frame.iloc[idx, 0]
        target_dir = os.path.join(self.image_dir, target)
        image = io.imread(target_dir)
        likes = self.landmarks_frame.iloc[idx, 1].astype('double')
        followers = self.landmarks_frame.iloc[idx, 2].astype('double')

        attractive = int(likes/followers*10)
        if attractive > 9:
            attractive = 9

        if self.transform:
            image = self.transform(image)

        return (image, attractive)


######################################################################
# Transforms
# ====================

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        # attractive = torch.LongTensor([attractive])
        return image


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class CropSquare(object):
    """Crop the image into a square

    """

    def __call__(self, image):
        
        h, w = image.shape[:2]

        if (h < w):
            new_w = h
            left = int(np.floor((w - new_w) / 2))
            img = image[0: h, left: left + new_w]

        else:
            new_h = w
            top = int(np.floor((h - new_h) / 2))
            img = image[top: top + new_h, 0: w]

        return img



######################################################################
# Testing
# ====================

import unittest
import matplotlib.pyplot as plt
import random
import time

class TestDataLoader(unittest.TestCase):

    def testDataLoader(self):

        plt.ion()   # interactive mode
        

        transformed_dataset = ChicagoFaceDatabase(root_dir='../data/ChicagoFaceDatabase/',
                                                  transform=transforms.Compose([
                                                      CropSquare(),
                                                      Rescale(256), ]))

        fig = plt.figure()
        for i in range(len(transformed_dataset)):
            n = random.randint(1,500)
            
            (image, attractive) = transformed_dataset[n]

            print(i, attractive,image.shape, image.size)

            ax = fig.add_subplot(2, 2, i + 1)
            im = ax.imshow(image)
            ax.set_title('Sample #{} attractive = {}'.format(i,attractive), fontsize = 8)
            ax.axis('off')
            plt.pause(0.001)

            if i == 3:
                plt.show()
                time.sleep(5)
                break


        transformed_dataset = ChicagoFaceDatabase(root_dir='../data/ChicagoFaceDatabase/',
                                                  transform=transforms.Compose([
                                                      CropSquare(),
                                                      Rescale(256),
                                                      ToTensor()]))
        for i in range(len(transformed_dataset)):
            (image, attractive) = transformed_dataset[i]

            print(i, attractive,image.shape, image.size)
            print(image)

            if i == 3:
                break

        # Here, image is a tensor, attractive is an int


        # test ins dataset loader
        ins_dataset = InstagramDatabase(root_dir='../data/InstagramSelfieDatabase',
                                                  transform=transforms.Compose([
                                                      CropSquare(),
                                                      Rescale(256), ]))

        fig = plt.figure()
        for i in range(len(ins_dataset)):
            n = random.randint(1,500)
            (image, attractive) = ins_dataset[n]

            print(i, attractive,image.shape, image.size)

            ax = fig.add_subplot(2, 2, i + 1)
            im = ax.imshow(image)
            ax.set_title('Sample #{} attractive = {}'.format(i,attractive), fontsize = 8)
            ax.axis('off')
            plt.pause(0.001)

            if i == 3:
                plt.show()
                time.sleep(5)
                break


if __name__ == '__main__':
    unittest.main()
