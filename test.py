"""

Sign language classifier

test.py

Testing script for sign language classifier.

"""
import torch
from torch.utils.data import DataLoader
from datasets import ASLAlphabet
from networks import SLClassifier
from argparse import ArgumentParser
from GradCAM import GradCAM

parser = ArgumentParser()
parser.add_argument('--filename',type=str,default="C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\sign_language_classifier\\trainlogs\\run2\\state_dicts\\net_state_dict_epoch7.pth",help="(str) filepath to network state dictionary")
parser.add_argument('--batchsize',type=int,default=2,help="Batch size")
parser.add_argument('--device',type=str,default='cuda',help="Device to compute on")
opts = parser.parse_args()

#
# Test network
#
dset = ASLAlphabet(type='val')
dl = DataLoader(dataset=dset,batch_size=opts.batchsize,shuffle=True)

net = SLClassifier()
net.load_state_dict(torch.load(opts.filename))
net.to(opts.device)
net.eval()

dlit = iter(dl)

# test accuracy
letters,samples = next(dlit)
scores = net(samples.to(opts.device))
_,predictions = scores.to('cpu').max(dim=1)
acc = (predictions==letters).sum() / scores.size(0)
print('acc =',acc)

# test grad-cam
GC = GradCAM(net,device=opts.device)
cams = GC(samples)
print('cams.shape =',cams.shape)
