"""

Sign language classifier

networks.py

Models and networks for sign language classifier.

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
        x = x.view(-1,1600)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.norm5(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.norm6(x)
        x = self.fc3(x)
        x = F.tanh(x)
        x = self.norm7(x)
        result = self.fc4(x) # raw network scores

        return result

"""
Run module as script.
"""
if __name__ == '__main__':
    from datasets import ASLAlphabet


    dset = ASLAlphabet() # get dataset
    net = SLClassifier() # initialize network

    samples = torch.cat([dset[0].unsqueeze(0),dset[1].unsqueeze(0)],dim=0)
    print('samples.shape =',samples.shape)

    scores = net(samples)
    print('scores.shape =',scores.shape)

    print("[ networks.py TESTING COMPLETE ]")
