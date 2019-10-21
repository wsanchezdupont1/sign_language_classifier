"""

Sign language classifier

test.py

Testing script for sign language classifier.

"""
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from datasets import ASLAlphabet
from networks import SLClassifier
from argparse import ArgumentParser
from GradCAM import GradCAM

import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--filename',type=str,default="C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\sign_language_classifier\\trainlogs\\run2\\state_dicts\\net_state_dict_epoch7.pth",help="(str) filepath to network state dictionary")
parser.add_argument('--batchsize',type=int,default=2,help="Batch size")
parser.add_argument('--ntestbatches',type=int,default=10,help="Number of test batches to evaluate accuracy on")
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

dliter = iter(dl) # dataloader iterator

# test accuracy
accs = torch.empty(opts.ntestbatches)
for i in range(opts.ntestbatches):
    letters,samples = next(dliter)
    scores = net(samples.to(opts.device))
    _,predictions = scores.to('cpu').max(dim=1)
    acc = (predictions==letters).sum() / scores.size(0)
    accs[i] = acc

print('accs = ',accs)

# test grad-cam
GC = GradCAM(net,device=opts.device)
cams = GC(samples).unsqueeze(2).detach() # add channel dim
print('cams.shape =',cams.shape)

imlist = [cam for cam in cams[0]]
print('len(imlist) =',len(imlist))
print('imlist[0].shape =',imlist[0].shape)

grid = make_grid(imlist,nrow=7,normalize=True).permute(1,2,0)
print('grid.shape =',grid.shape)

plt.imshow(grid)
plt.show()
