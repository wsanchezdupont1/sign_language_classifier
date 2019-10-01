"""

Sign language classifier

train.py

Training script for sign language classifier.

"""
def train(net,optimizer,lossfunc,dataloader,batchsize=32,numepochs=100,device='cuda'):
    """
    train

    Train a sign language classifier network on the asl-alphabet.

    inputs:
        net - (SLClassifier) Sign language classifier network
        optimizer - (torch.optim optimizer) Optimizer object created using net's parameters
        dataloader - (ASLAlphabet) ASLAlphabet dataloader
        batchsize - (int) number of samples per batches
        numepochs - (int) number of epochs to train on
        device - (str) device to perform computations on

    """
    net.to(device) # move params to device

    for epoch in range(numepochs):
        for i,batch in enumerate(dataloader):
            labels,samples = batch
            print('labels.shape =',labels.shape)
            print('samples.shape =',samples.shape)
            scores = net(samples.to(device))
            print('scores.shape =',scores.shape)

    return

"""

Run module as script.

1. Collect hyperparameter settings via ArgumentParser
2. Initialize dataset + network + optimizer
3. Use train function to train the network for given number of batches

"""
if __name__ == "__main__":
    from argparse import ArgumentParser
    import datasets
    import networks
    import torch
    from torch.utils.data import DataLoader

    parser = ArgumentParser()

    # misc but important arguments
    parser.add_argument('-n','--numepochs',type=int,default=1,help='(int) Number of epochs to train on | default: 1 (for testing only)')
    parser.add_argument('-b','--batchsize',type=int,default=32,help='(int) Number of samples per batch | default: 32')
    parser.add_argument('-d','--device',type=str,default='cuda',help='(str) Device to process on | default: cuda')
    # parser.add_argument('--lossType',type=str,default='crossentropy',help='(str) Loss type | default: crossentropy')

    # optimizer args
    # parser.add_argument('--optimizertype',type=str,default='SGD',help='(str) Optimizer type | default:SGD')
    parser.add_argument('--lr',type=float,default=0.001,help='(float) Learning rate | default: 0.001')

    # add more args here as features are added...

    # parse
    opts = parser.parse_args()

    # initialize network and optimizer
    net = networks.SLClassifier() # can add network configs here later
    optimizer = torch.optim.SGD(net.parameters(),lr=opts.lr) # SGD for now, add options later

    # initialize dataloader
    dataloader = DataLoader(datasets.ASLAlphabet(train=True),batch_size=opts.batchsize,shuffle=True)

    # initialize loss function
    lossfunc = torch.nn.CrossEntropyLoss()

    train(net,optimizer,lossfunc,dataloader,batchsize=opts.batchsize,numepochs=opts.numepochs,device=opts.device)
