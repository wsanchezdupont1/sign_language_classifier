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
    Works for training and validation.
    """
    def __init__(self,basedir='C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\asl_alphabet',type='train'):
        """
        __init__

        Constructor.

        inputs:
            basedir - (str) base directory of dataset
            type - (str) type of datset ('train' or 'val')
        """
        super(ASLAlphabet,self).__init__()

        self.basedir = basedir
        if type == 'train':
            self.basedir = os.path.join(self.basedir,'train')
        elif type == 'val':
            self.basedir = os.path.join(self.basedir,'val')
        else:
            raise Exception('Invalid dataset type!')


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

class TestASLAlphabet(Dataset):
    """
    TestASLAlphabet

    A test class for the ASL alphabet dataset. Link: https://www.kaggle.com/grassknoted/asl-alphabet
    """
    def __init__(self,basedir="C:/Users/Willis/Desktop/Sign Language Classifier/asl_alphabet/test"):
        """
        __init__

        Constructor.

        inputs:
            basedir - (str) direcory where train/val images are stored
        """
        super(TestASLAlphabet,self).__init__()

        self.basedir = basedir
        self.imnames = os.listdir(self.basedir)
        self.imnames.sort()

        # get list of letters
        self.letters = ''
        for imname in self.imnames:
            self.letters += imname
        self.letters = self.letters.split('_test.jpg')[:-1]

    def __len__(self):
        """
        __len__

        Number of samples in the dataset.

        outputs:
            length - (int) number of samples in the dataset
        """
        return len(self.imnames)

    def __getitem__(self,idx):
        """
        __getitem__

        Get the image at index idx.

        inputs:
            idx - (int) index of test image

        outputs:
            x - (Tensor) torch.Tensor image sample
        """
        filename = self.imnames[idx]
        letter = self.letters.index(filename.split('_test.jpg')[0])
        x = ToTensor()(Image.open(os.path.join(self.basedir, filename))) # open and make tensor
        return letter,x

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
    letter,sample = dset[len(dset)-1]
    print('letter =',letter)
    print('sample.shape =',sample.shape)
    plt.imshow(np.array(sample.permute(1,2,0))) # move channels to 3rd dim and imshow
    plt.show()
    print('[ train dataset tests complete ]')

    dset = TestASLAlphabet()
    letter,sample = dset[len(dset)-1]
    print('letter =',letter)
    print('sample.shape =',sample.shape)
    plt.imshow(np.array(sample.permute(1,2,0))) # move channels to 3rd dim and imshow
    plt.show()
    print('[ test dataset tests complete]')

    print('[ datasets.py TESTING COMPLETE ]')
