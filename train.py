"""

Sign language classifier

train.py

Training script for sign language classifier.

"""
# custom imports
import datasets
import networks

# pytorch stuff
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# misc imports
import os
from argparse import ArgumentParser
import time

def train(net,
          optimizer,
          lossfunc,
          train_dataloader,
          val_dataloader,
          batchsize=32,
          numepochs=1,
          device='cuda',
          log_basedir='/home/wjsd/Desktop/Coding projects/sign_language_classifier/sign_language_classifier/trainlogs',
          log_subdir='run0',
          log_frequency=100,
          val_frequency=100,
          ):
    """
    train

    Train a sign language classifier network on the asl-alphabet.

    inputs:
        net - (SLClassifier) Sign language classifier network
        optimizer - (torch.optim optimizer) Optimizer object created using net's parameters
        train_dataloader - (ASLAlphabet) ASLAlphabet training dataloader
        val_dataloader - (ASLAlphabet) ASLAlphabet validation dataloader
        batchsize - (int) number of samples per batches
        numepochs - (int) number of epochs to train on
        device - (str) device to perform computations on
        log_basedir - (str) project logging folder that holds all logs (e.g. "C:\\Users...\\project_name\logs")
        log_subdir - (str) subdirectory of log_basedir specifying the storage folder for _this_ experiment (e.g. "run1")
        log_frequency - (int) logging frequency (in number of batches)
        val_frequency - (int) process validation batch every val_frequency samples

    """

    net.to(device) # move params to device
    print('[ network pushed to device ]')

    logpath = os.path.join(log_basedir,log_subdir)
    s = SummaryWriter(log_dir=os.path.join(logpath,'training')) # start tensorboard writer

    # create state_dict log folder
    state_dict_path = os.path.join(logpath,'state_dicts')
    os.mkdir(state_dict_path)


    # TODO: use real image
    # load graph of net with dummy input image
    s.add_graph(net,torch.rand(1,3,200,200).to(device))

    print('[ starting training ]')
    print('----------------------------------------------------------------')
    t_start = time.time() # record

    if val_frequency is not None:
        val_dataloader_it = iter(val_dataloader) # use this to load validation batches when we want
        v = SummaryWriter(log_dir=os.path.join(logpath,'validation'))

    batches_processed = 0
    logstep = 0 # the "global_step" variable for tensorboard logging
    for epoch in range(numepochs):
        for i,batch in enumerate(train_dataloader):
            # start device transfer timing
            transfer_start = time.time()
            # sample and move to device
            labels,samples = batch
            samples = samples.to(device)
            labels = labels.to(device)
            transfer_time = time.time() - transfer_start # record cpu dataload time

            # gpu computations
            compute_start = time.time()
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

            compute_time = time.time() - compute_start

            batches_processed += 1

            #
            # Tensorboard logging
            #
            if batches_processed % log_frequency == 0:
                print('logstep =',logstep)
                print('batches_processed =',batches_processed)
                print('epoch_progress =',batchsize*batches_processed/(2550*29)) # TODO: remove hardcoded 2550 samples/class * 29 classes
                print('samples_processed =',batchsize*batches_processed)

                # TODO: make sure this is working fully
                # log time for log_freq batches
                t_end = time.time()
                s.add_scalars('times',{'batches_{}'.format(log_frequency):t_end-t_start, 'transfer_time':transfer_time, 'compute_time':compute_time},logstep)
                t_start = t_end

                # compute accuracy
                _,class_pred = probs.max(dim=1)
                acc = (class_pred==labels).sum() / float(len(labels))  # accuracy
                s.add_scalars('accuracies',{'train':acc},logstep)

                # record batch loss mean + std
                s.add_scalars('losses',{'loss/mean':loss_mean,'loss/var':loss_var},logstep)

                # histograms
                # TODO: add histogram of weights
                s.add_histogram('losses/batch',loss,logstep) # batch losses

                # TODO: get PR curves working
                one_hot = torch.nn.functional.one_hot(labels)
                # sometimes 29th class isn't represented, so one_hot results in <29 columns
                if one_hot.size(1) < 29:
                    one_hot = torch.cat([one_hot.cpu(), torch.zeros(one_hot.size(0), 29-one_hot.size(1)).long()],dim=1)

                s.add_pr_curve('pr',labels=one_hot,predictions=probs,global_step=logstep)

                # gpu usage
                # TODO: optimize gpu usage
                s.add_scalars('gpu_usage',{'mem_allocated':torch.cuda.memory_allocated('cuda'),'mem_cached':torch.cuda.memory_cached('cuda')},logstep)

                # TODO: sample images
                # s.add_image('batch_samples',torch.make_grid())

                # TODO: grad-CAM

                # TODO: activation distributions for layers

                logstep += 1
                print('----------------------------------------------------------------')

            #
            # Validation
            #
            # TODO: prevent validation StopIteration from eventually happening with DataLoader on long training runs
            # TODO: finish validation
            if val_frequency != 0:
                if batches_processed % val_frequency == 0:
                    with torch.no_grad():
                        labels,samples = next(val_dataloader_it)
                        labels = labels.to(device)
                        samples = samples.to(device)

                        scores = net(samples)
                        probs = scores.softmax(dim=1)

                        # val losses
                        loss_val = lossfunc(scores,labels)
                        loss_val_mean = loss_val.mean()
                        loss_val_var = loss_val.var()
                        v.add_scalars('losses',{'val_mean':loss_val_mean,'val_var':loss_val_var},logstep)

                        # val accuracy
                        _,class_pred = probs.max(dim=1)
                        val_acc = (class_pred==labels).sum() / float(len(labels))  # accuracy
                        v.add_scalars('accuracies',{'validation':val_acc},logstep)

        # checkpoint model every epoch
        pth_path = os.path.join(state_dict_path, 'net_state_dict_{}{}.pth'.format('epoch',epoch))
        torch.save(net.state_dict(),pth_path)
        print('[ model saved, path = %s ]' % pth_path)

        # TODO: add validation


    return net # return the network for subsequent usage


def saveopts(opts_filename,opts):
    """
    saveopts

    Helper function to save parsed arguments. Saves to .txt file with the
    following format:

    arg1: value1
    arg2: value2
    ...
    argN: valueN

    inputs:
        opts_filename - (str) directory inside which to save options
        opts - (argparse.Namespace) object containing parsed options (output of parser.parse_args())
    """
    with open(opts_filename,'w') as f:
        for arg,val in vars(opts).items():
            line = '{}: {}\n'.format(arg,val)
            f.write(line)

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
    parser.add_argument('--val_batchsize',type=str,default=32,help='(int) Validation dataloader batch size | default: 32')
    parser.add_argument('-d','--device',type=str,default='cuda',help='(str) Device to process on | default: cuda')
    parser.add_argument('--log_basedir',type=str,default="/home/wjsd/Desktop/Coding projects/sign_language_classifier/sign_language_classifier/trainlogs/",help="(str) Project logging folder that holds all logs (e.g. 'C:\\Users...\\project_name\logs')")
    parser.add_argument('--log_subdir',type=str,default='run0',help="(str) Subdirectory of log_basedir specifying the storage folder for this particular experiment (e.g. 'run1')")
    parser.add_argument('-lf','--log_freq',type=int,default=100,help="(int) Logging frequency (number of batches")
    parser.add_argument('-vf','--val_frequency',type=int,default=100,help="(int) Validation batch frequency")
    parser.add_argument('--num_workers',type=int,default=0,help="(int) Number of DataLoader cpu workers | default: 0")
    # parser.add_argument('--lossType',type=str,default='crossentropy',help='(str) Loss type | default: crossentropy')
    parser.add_argument('--lr',type=float,default=0.001,help='(float) Learning rate | default: 0.001')
    parser.add_argument('--lossfunc',type=str,default='crossentropy',help="(str) Loss functiion type | default: 'crossentropy'")
    # parser.add_argument('--optimizertype',type=str,default='SGD',help='(str) Optimizer type | default:SGD')

    # add more args here as features are added...

    # parse
    opts = parser.parse_args()

    # clear out log_subdir
    logpath = os.path.join(opts.log_basedir,opts.log_subdir)
    if os.path.exists(logpath):
        import shutil
        shutil.rmtree(logpath)
    os.mkdir(logpath)

    # save parsed arguments to txt file
    opts_filename = os.path.join(logpath,'opts_{}.pth'.format(opts.log_subdir))
    print('saving parsed arguments to .txt file: {}'.format(opts_filename))
    saveopts(opts_filename,opts)

    # initialize network and optimizer
    # TODO: add  weight initialization!!!!
    net = networks.SLClassifier() # TODO: add network configs here
    optimizer = torch.optim.SGD(net.parameters(),lr=opts.lr) # SGD for now, add options later

    # initialize dataloader
    dataset = datasets.ASLAlphabet(type='train')
    train_dataloader = DataLoader(dataset,batch_size=opts.batchsize,shuffle=True,num_workers=opts.num_workers)
    dataset = datasets.ASLAlphabet(type='val')
    val_dataloader = DataLoader(dataset,batch_size=opts.val_batchsize,shuffle=True,num_workers=opts.num_workers)

    # initialize loss function
    if opts.lossfunc in ['crossentropy','crossentropyloss']:
        lossfunc = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        raise Exception("Loss function invalid!")

    train(net,
          optimizer,
          lossfunc,
          train_dataloader,
          val_dataloader,
          batchsize=opts.batchsize,
          numepochs=opts.numepochs,
          device=opts.device,
          log_basedir=opts.log_basedir,
          log_subdir=opts.log_subdir,
          log_frequency=opts.log_freq,
          val_frequency=opts.val_frequency)

    print('[ testing eval mode ... ]')
    net.cpu()
    net.eval()
    letter,sample = dataset[0]
    probs = net(sample.unsqueeze(0)).softmax(dim=1)
    print('probs =',probs)
