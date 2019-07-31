"""

Sign language classifier

datasets.py

Dataset file for sign language classifer. Uses Pytorch.

Notes:



"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import math
from torchvision.transforms import ToTensor,Normalize

class ASLAlphabet(Dataset):
    """
    ASLAlphabet

    A dataset class for a Kaggle ASL alphabet dataset. Link: https://www.kaggle.com/grassknoted/asl-alphabet
    """
    def __init__(self,basedir='C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\asl_alphabet',train=True):
        """
        __init__

        Constructor.

        inputs:
            basedir - (str) base directory of dataset
            train - (bool) training set if true, test set if false.
        """
        super(ASLAlphabet,self).__init__()

        self.basedir = basedir
        if train:
            self.basedir = os.path.join(self.basedir,'train')
        else:
            self.basedir = os.path.join(self.basedir,'test')

        self.dirnames = os.listdir(self.basedir) # get names of each class directory
        dset.dirnames.sort()

    def __len__(self):
        """
        __len__

        Returns number of samples in dataset.
        """
        return 3000*29 # 29 classes * 3000 images per class

    def  __getitem__(self,idx):
        """
        __getitem__

        Extract dataset item at index idx.

        inputs:
            idx - (int) index value of desired sample

        outputs:
            x - (torch.Tensor) sample idx from dataset
        """
        letter = math.floor(idx/3000) # index of folder within train directory
        index = idx%3000 # index within letter directory
        filename = os.path.join(self.basedir, self.dirnames[letter])

        x = ToTensor()(Image.open(os.path.join(filename, os.listdir(filename)[index]))) # open and make tensor
        x = Normalize(x.mean(dim=))
        """
        Left off here
        """

        return x


"""
Module testing
"""
if __name__ == "__main__":
    dset = ASLAlphabet()

    print(dset.dirnames)
