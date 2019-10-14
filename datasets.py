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
from torchvision.transforms import ToTensor

class ASLAlphabet(Dataset):
    """
    ASLAlphabet

    A dataset class for a Kaggle ASL alphabet dataset. Link: https://www.kaggle.com/grassknoted/asl-alphabet.
    """
    def __init__(self,basedir='C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\asl_alphabet\\train'):
        """
        __init__

        Constructor.

        inputs:
            basedir - (str) base directory of dataset
        """
        super(ASLAlphabet,self).__init__()

        self.basedir = basedir
        self.dirnames = os.listdir(self.basedir) # get names of each class directory
        self.dirnames.sort()
        self.samplesPerClass = len(os.listdir(os.path.join(self.basedir, self.dirnames[0])))

    def __len__(self):
        """
        __len__

        Returns number of samples in dataset.
        """
        return self.samplesPerClass*29

    def  __getitem__(self,idx):
        """
        __getitem__

        Extract dataset item at index idx.

        inputs:
            idx - (int) index value of desired sample

        outputs:
            x - (torch.Tensor) sample idx from dataset
            letter - (str) letter/class that x is from
        """
        letter = math.floor(idx/self.samplesPerClass) # index of folder within train directory
        index = idx%self.samplesPerClass # index within letter directory
        filename = os.path.join(self.basedir, self.dirnames[letter])

        x = ToTensor()(Image.open(os.path.join(filename, os.listdir(filename)[index]))) # open and make tensor
        return letter,x # don't normalize here, use norm as first network layer


"""
Module testing
"""
if __name__ == "__main__":
    #
    # constructor test
    #
    dset = ASLAlphabet()
    print('dset.dirnames =',dset.dirnames)

    #
    # imshow test
    #
    import numpy as np
    from matplotlib import pyplot as plt
    letter,sample = dset[dset.samplesPerClass*29-1]
    print('letter =',letter)
    print('sample.shape =',sample.shape)
    plt.imshow(np.array(sample.permute(1,2,0))) # move channels to 3rd dim and imshow
    plt.show()

    print('[ datasets.py TESTING COMPLETE ]')
