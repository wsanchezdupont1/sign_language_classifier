"""

Sign language classifier

train.py

Training script for sign language classifier.

"""
import os
from argparse import ArgumentParser
import datasets
import networks
import torch
from torch.utils.data import DataLoader

def train(net,
          optimizer,
          lossfunc,
          dataloader,
          batchsize=32,
          numepochs=1,
          device='cuda',
          log_basedir='C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\sign_language_classifier\\trainlogs',
          logdir='run1',
          log_frequency=100
          ):
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
        log_basedir - (str) project logging folder that holds all logs (e.g. "C:\\Users...\\project_name\logs")
        logdir - (str) subdirectory of log_basedir specifying the storage folder for _this_ experiment (e.g. "run1")
        log_frequency - (int) logging frequency (in number of batches)

    """
    from torch.utils.tensorboard import SummaryWriter
    import time

    net.to(device) # move params to device
    print('[ network pushed to device ]')

    logpath = os.path.join(log_basedir,logdir)
    s = SummaryWriter(log_dir=logpath) # start tensorboard writer

    # create state_dict log folder if it doesn't exist
    state_dict_path = os.path.join(logpath,'state_dicts')
    if not os.path.exists(state_dict_path):
        os.mkdir(state_dict_path)


    # load graph of net with dummy input image
    s.add_graph(net,torch.rand(1,3,200,200).to(device))

    print('[ starting training ]')
    print('----------------------------------------------------------------')
    t_start = time.time() # record

    batches_processed = 0
    logstep = 0 # the "global_step" variable for tensorboard logging
    for epoch in range(numepochs):
        for i,batch in enumerate(dataloader):

            # TODO: add validation

            # sample and move to device
            labels,samples = batch
            samples = samples.to(device)
            labels = labels.to(device)

            scores = net(samples)
            probs = scores.softmax(dim=1)

            loss = lossfunc(scores,labels) # with batch dim (note: default reduction is 'none' - see below)

            # TODO: add regularization (e.g. L1/L2)

            # store mean and std
            loss_var = torch.var(loss)
            loss_mean = torch.mean(loss)

            # backprop + paramater update
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            batches_processed += 1

            #
            # Tensorboard logging
            #
            if batches_processed % log_frequency == 0:
                print('logstep =',logstep)
                print('batches_processed =',batches_processed)

                # log time for 100 batches
                t_end = time.time()
                s.add_scalar('times_100batch',t_end-t_start,logstep)
                t_start = t_end

                # compute accuracy
                _,class_pred = probs.max(dim=1)
                acc = (class_pred==labels).sum() / float(len(labels))  # accuracy

                # record batch loss mean + std
                s.add_scalars('run1/losses',{'loss/mean':loss_mean,'loss/var':loss_var},logstep)
                s.add_scalar('accuracy',acc,logstep)

                # histograms
                # TODO: add histogram of weights
                s.add_histogram('run1/losses/batch',loss,logstep) # batch losses
                # TODO: add  weight initialization!!!!

                # TODO: precision-recall curve
                print('labels.shape =',labels.shape)
                print('probs.shape =',probs.shape)
                one_hot = torch.nn.functional.one_hot(labels)
                print('one_hot.shape =',one_hot.shape)

                # sometimes 29th class isn't represented, so one_hot results in <29 columns
                if one_hot.size(1) < 29:
                    one_hot = torch.cat([one_hot.cpu(), torch.zeros(one_hot.size(0), 29-one_hot.size(1)).long()],dim=1)

                # TODO: fix this!!
                s.add_pr_curve('run1',labels=one_hot,predictions=probs,global_step=logstep)

                # TODO: sample images
                # s.add_image('batch_samples',torch.make_grid())

                # TODO: grad-CAM

                # TODO: activation distributions for layers

                logstep += 1
                print('----------------------------------------------------------------')

        # checkpoint model every epoch
        pth_path = os.path.join(state_dict_path, 'net_state_dict_%s%i.pth' % ('epoch',epoch))
        torch.save(net.state_dict(),pth_path)
        print('[ model saved, path = %s ]' % pth_path)


    return net # return the network for subsequent usage

"""

Run module as script.

1. Collect hyperparameter settings via ArgumentParser
2. Initialize dataset + network + optimizer
3. Use train function to train the network for given number of batches

"""
if __name__ == "__main__":

    parser = ArgumentParser()

    # misc but important arguments
    parser.add_argument('-n','--numepochs',type=int,default=1,help='(int) Number of epochs to train on | default: 1 (for testing only)')
    parser.add_argument('-b','--batchsize',type=int,default=32,help='(int) Number of samples per batch | default: 32')
    parser.add_argument('-d','--device',type=str,default='cuda',help='(str) Device to process on | default: cuda')
    parser.add_argument('--log_basedir',type=str,default="C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\sign_language_classifier\\trainlogs",help="(str) project logging folder that holds all logs (e.g. 'C:\\Users...\\project_name\logs')")
    parser.add_argument('--logdir',type=str,default="run1",help="(str) subdirectory of log_basedir specifying the storage folder for this particular experiment (e.g. 'run1')")
    parser.add_argument('-lf','--log_freq',type=int,default=100,help="(int) logging frequency (number of batches")
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
    dataset = datasets.ASLAlphabet(train=True)
    dataloader = DataLoader(dataset,batch_size=opts.batchsize,shuffle=True)

    # initialize loss function
    lossfunc = torch.nn.CrossEntropyLoss(reduction='none')

    train(net,
          optimizer,
          lossfunc,
          dataloader,
          batchsize=opts.batchsize,
          numepochs=opts.numepochs,
          device=opts.device,
          log_basedir=opts.log_basedir,
          logdir=opts.logdir,
          log_frequency=opts.log_freq)

    print('[ testing eval mode ... ]')
    net.eval()
    sample = dataset[0].unsqueeze(0)
    probs = net(sample).softmax(dim=1)
    print('probs =',probs)
