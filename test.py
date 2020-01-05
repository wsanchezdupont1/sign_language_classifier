"""

Sign language classifier

test.py

Testing script for sign language classifier.

TODO: make sure printed letter is correct

"""
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from datasets import ASLAlphabet
from networks import SLClassifier
from argparse import ArgumentParser
from GradCAM import GradCAM
import cv2

import matplotlib.pyplot as plt
import os

parser = ArgumentParser()
parser.add_argument('--filename',type=str,default="/home/wjsd/Desktop/Coding projects/sign_language_classifier/sign_language_classifier/trainlogs/run2/state_dicts/net_state_dict_epoch7.pth",help="(str) filepath to network state dictionary")
parser.add_argument('--batchsize',type=int,default=36,help="Batch size")
parser.add_argument('--ntestbatches',type=int,default=10,help="Number of test batches to evaluate accuracy on")
# TODO: cams by batch
parser.add_argument('-cbb','--cams_by_batch',action='store_true',help="If flagged, make image grid by batch rather than by letter")
parser.add_argument('--device',type=str,default='cuda',help="Device to compute on")
opts = parser.parse_args()

#
# Test network
#
dset = ASLAlphabet(type='val')
dl = DataLoader(dataset=dset,batch_size=opts.batchsize,shuffle=True)
print('[ dataloader created ]')

net = SLClassifier()
net.load_state_dict(torch.load(opts.filename))
net.to(opts.device)
net.eval()
print('[ state_dict loaded into network ]')

dliter = iter(dl) # dataloader iterator

# test accuracy
accs = torch.empty(opts.ntestbatches)
for i in range(opts.ntestbatches):
    letters,samples = next(dliter)
    scores = net(samples.to(opts.device))
    _,predictions = scores.to('cpu').max(dim=1)
    acc = (predictions==letters).sum() / (scores.size(0)*1.0)
    accs[i] = acc

print('predictions =',predictions)
print('letters =',letters)
print('acc =',acc)
print('(predictions==letter) =',(predictions==letters))

# TODO: fix accuracies
print('accs = ',accs)

# test grad-cam
GC = GradCAM(net,device=opts.device)
samples.requires_grad = True
cams = GC(samples,submodule=net.conv6,guided=opts.guided).unsqueeze(2).detach() # add channel dim
print('cams.shape =',cams.shape)

cams = torch.nn.Upsample(scale_factor=2,mode='bilinear')(cams[0])
print('cams.shape =',cams.shape)

imlist = [cam for cam in cams] # convert first dim into list
dirs = os.listdir('/home/wjsd/Desktop/Coding projects/sign_language_classifier/asl_alphabet/train')
print('dirs =',dirs)
dirs.sort()
print('sorted dirs =',dirs)
print('letter =',dirs[letters[0]])
print('prediction =',dirs[predictions[0]])
print('len(imlist) =',len(imlist))
print('imlist[0].shape =',imlist[0].shape)

grid = make_grid(imlist,nrow=6,normalize=True).permute(1,2,0)
print('grid.shape =',grid.shape)

fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1)
ax1.imshow(samples[0].permute(1,2,0))
ax2.imshow(grid)

plt.show()
