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
        self.norm1 = nn.BatchNorm2d(num_features=3) # 3-channel inputs
        self.conv1 = nn.Conv2d(3,64,4) # output shape (197,197)
        self.conv2 = nn.Conv2d(64,64,4) # output shape (194,194)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(64,128,4,stride=2) # output shape (96,96)
        self.conv4 = nn.Conv2d(128,128,4) # output shape (93,93)
        self.conv5 = nn.Conv2d(128,64,4) # output shape (90,90)
        self.norm3 = nn.BatchNorm2d(num_features=64)
        self.conv6 = nn.Conv2d(64,32,4,stride=2) # output shape (44,44)
        self.conv7 = nn.Conv2d(32,16,4,stride=2) # output shape (21,21)
        self.conv8 = nn.Conv2d(16,16,3,stride=2) # output shape (10,10)
        self.norm4 = nn.BatchNorm2d(num_features=16)

        self.flat = Flatten()

        self.fc1 = nn.Linear(1600,800)
        self.norm5 = nn.BatchNorm1d(num_features=800)
        self.fc2 = nn.Linear(800,400)
        self.norm6 = nn.BatchNorm1d(num_features=400)
        self.fc3 = nn.Linear(400,100)
        self.norm7 = nn.BatchNorm1d(num_features=100)
        self.fc4 = nn.Linear(100,29)

    def forward(self,x):
        """
        forward

        Forward pass.

        inputs:
            x - (torch.Tensor) tensor image input

        outputs:
            result - (torch.Tensor) tensor output of the network
        """
        # encode
        x = self.norm1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.norm3(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.norm4(x)

        # reshape
        x = self.flat(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.norm5(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.norm6(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.norm7(x)
        result = self.fc4(x) # raw network scores

        return result

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
