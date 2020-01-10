"""

Sign language classifier

networks.py

Models and networks for sign language classifier.

Notes:

TODO: weight initialization

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SLClassifier(nn.Module):
    """
    SLClassifier

    A sign language classifier network.
    """
    def __init__(self):
        """
        __init__

        Constructor.
        """
        super(SLClassifier,self).__init__()

        #
        # Image feature encoder
        #
        # bnorm goes first here since dataset samples are unnormalized (also enables normalization using gpu)
        #
        self.features = torch.nn.ModuleList([])
        self.features.append(nn.BatchNorm2d(num_features=3)) # 3-channel inputs

        self.features.append(nn.Conv2d(3,32,3))
        self.features.append(nn.Conv2d(32,32,3)) # Nx32x198x198
        self.features.append(nn.BatchNorm2d(num_features=32)) # Nx32x196x196

        self.features.append(nn.Conv2d(32,64,4,stride=2)) # Nx64x97x97
        self.features.append(nn.Conv2d(64,64,4)) # Nx64x94x94
        self.features.append(nn.BatchNorm2d(num_features=64)) # Nx64x94x94

        self.features.append(nn.Conv2d(64,64,4,stride=2)) # Nx64x46x46
        self.features.append(nn.Conv2d(64,64,4)) # Nx64x43x43
        self.features.append(nn.BatchNorm2d(num_features=64)) # Nx64x43x43

        self.features.append(nn.Conv2d(64,32,3,stride=2)) # Nx32x21x21
        self.features.append(nn.Conv2d(32,16,4)) # Nx16x18x18
        self.features.append(nn.BatchNorm2d(num_features=16)) # Nx16x18x18

        self.classifier = torch.nn.ModuleList([])
        self.classifier.append(Flatten()) # Nx5184 = Nx16*18*18
        self.classifier.append(nn.Linear(5184,256))
        self.classifier.append(nn.Tanh())
        self.classifier.append(nn.Linear(256,128))
        self.classifier.append(nn.Tanh())
        self.classifier.append(nn.Linear(128,29))

    def forward(self,x):
        """
        forward

        Forward pass.

        inputs:
            x - (torch.Tensor) tensor image input

        outputs:
            x - (torch.Tensor) tensor output of the network
        """
        for i in range(len(self.features)):
            x = self.features[i](x)

        for i in range(len(self.classifier)):
            x = self.classifier[i](x)

        return x

class Flatten(torch.nn.Module):
    """
    Flatten

    torch.view as a layer.
    """
    def __init__(self,start_dim=1):
        """
        __init__

        Constructor.

        inputs:
        start_idm - (bool) dimension to begin flattening at.
        """
        super(Flatten,self).__init__()
        self.start_dim = start_dim

    def forward(self,x):
        """
        forward

        Forward pass.
        """
        return x.flatten(self.start_dim)

"""
Run module as script (for testing, mostly).
"""
if __name__ == '__main__':
    from datasets import ASLAlphabet
    from matplotlib import pyplot as plt

    dset = ASLAlphabet() # get dataset
    net = SLClassifier() # initialize network

    samples = torch.cat([dset[dset.samplesPerClass*29-1][1].unsqueeze(0), dset[0][1].unsqueeze(0)],dim=0)
    print('samples.shape =',samples.shape)

    # plot
    plt.imshow(samples[0].permute(1,2,0))
    plt.show()

    scores = net(samples)
    print('scores.shape =',scores.shape)

    # make sure batch norm works during eval
    net.eval()
    letter,sample = dset[0]
    score = net(sample.unsqueeze(0))
    print('score.shape =',score.shape,'(.eval() check)')

    print("[ networks.py TESTING COMPLETE ]")
