"""

Willis Sanchez-duPont

Implementation of Grad-CAM (https://arxiv.org/abs/1610.02391).

Notes:

- (batch summing) In order to be able to process a batch of examples at once,
  we can sum out the batch dimension of the model output and then call .backward()
  on each class, rather than call the .backward() function  B samples * C classes
  number of times. The gradient of the class output w.r.t each activation map pixel
  is unchanged by this operation (i.e. in backprop we take each output and pass
  it thru a sum operator, and the local gradient of the sum operation w.r.t. its
  inputs is one, so the chain rule leaves dC_i/da_abcd intact where a,b,c, and d
  are indices for batch, channel, row, and column respectively and C_i is class i)

"""

import torch
import torch.nn as nn

class GradCAM():
    """
    GradCAM

    Class that performs grad-CAM algorithm on a given model (ARXIV LINK HERE).
    """
    def __init__(self,model,device='cuda'):
        """
        Constructor.

        inputs:
            model - (torch.nn.Module) Indexable module to perform grad-cam on (e.g. torch.nn.Sequential)
            device - (str) device to perform computations on
        """
        self.device = device
        self.inputs = torch.empty(0)
        self.model = model
        model.to(self.device) # push params to device

        self.activations = torch.empty(0) # initialize activation maps
        self.grads = torch.empty(0) # initialize gradient maps
        self.results = torch.empty(0) # network output holder
        self.gradCAMs = torch.empty(0) # output maps
        self.fh = [] # module forward hook
        self.bh = [] # module backward hook

    def __call__(self,x,submodule=None,classes='all'):
        """
        __call__

        Compute class activation maps for a given input and submodule.

        inputs:
            x - (torch.Tensor) input image to compute CAMs for.
            submodule - (str or torch.nn.Module) name of module to get CAMs from OR submodule object to hook directly
            classes - (list) list of indices specifying which classes to compute grad-CAM for (MUST CONTAIN UNIQUE ELEMENTS). CAMs are returned in same order as specified in classes.
        """
        self.inputs = x.to(self.device)

        if submodule is None:
            self.activations = self.inputs # treat inputs as activation maps

        # hook registration
        self.sethooks(submodule)

        # forward pass
        self.results = self.model(self.inputs) # store result (already on device)
        self.gradCAMs = torch.empty(0)

        # grad and CAM computations
        summed = self.results.sum(dim=0) # sum out batch results. See note at top of file

        # retain graph if not on last iteration
        if classes == 'all':
            classes = [x for x in range(self.results.size(1))] # list of all classes
        for c in classes: # loop thru classes
            if c == classes[-1]:
                summed[c].backward()
            else:
                summed[c].backward(retain_graph=True) # retain graph to be able to backprop without calling forward again

            # see paper for details on math
            coeffs = self.grads.sum(-1).sum(-1) / (self.grads.size(2) * self.grads.size(3)) # coeffs has size = (batchsize, activations.shape(1)
            prods = coeffs.unsqueeze(-1).unsqueeze(-1)*self.activations # align dims to get appropriate linear combination of feature maps (which reside in dim=1 of self.activations)
            cam = torch.nn.ReLU()(prods.sum(dim=1)) # sum along activation dimensions (result size = batchsize x U x V)
            self.gradCAMs = torch.cat([self.gradCAMs, cam.unsqueeze(0).to('cpu')],dim=0) # add CAMs to function output variable
            self.model.zero_grad() # clear gradients for next backprop


        self.bh.remove()
        if submodule is not None:
            self.fh.remove()

        return self.gradCAMs.permute(1,0,2,3)

    def sethooks(self,submodule=None):
        """
        sethooks

        Set hooks for each layer.

        inputs:
            submodule - (str or torch.nn.Module) name of submodule within self.model to hook OR submodule to hook directly
        """
        # define hook functions
        def getactivation(mod,input,output):
            """
            getactivation

            Copy activations to self.activations on forward pass.

            inputs:
                mod - (torch.nn.Module) module being hooked
                input - (tuple) inputs to submodule being hooked (ignore)
                output - (tensor) output of submodule being hooked (NOTE: untested for modules with multiple outputs)
            """
            self.activations = output

        def getgrad(mod,gradin,gradout):
            """
            getgrad

            Get gradients during hook.

            inputs:
                mod - (torch.nn.Module) module being hooked
                gradin - (tuple) inputs to last operation in mod (ignore)
                gradout - (tuple) gradients of class activation w.r.t. outputs of mod
            """
            self.grads = gradout[0]

        def getgrad_input(g):
            """
            getgrad_input

            Get gradient on backward pass when using inputs for CAMs.

            inputs:
                g - (torch.Tensor) tensor to set/store as gradient on backward pass
            """
            self.grads = g


        if submodule is None: # if using inputs as activation maps
            self.activations.requires_grad=True # make sure input grads are available
            self.bh = self.activations.register_hook(getgrad_input) # save gradient
        elif type(submodule) == type(""): # if using string name of submodule...
            for name,module in self.model._modules.items():
                if name == submodule:
                    self.fh = module.register_forward_hook(getactivation) # forward hook
                    self.bh = module.register_backward_hook(getgrad) # backward hook
        else:
            self.fh = submodule.register_forward_hook(getactivation) # forward hook
            self.bh = submodule.register_backward_hook(getgrad) # backward hook


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
MODULE TESTING
"""
if __name__ == '__main__':
    '''
    Imports
    '''
    from matplotlib import pyplot as plt
    import numpy as np

    from torchvision import models

    '''
    Basic test with dummy inputs/model.
    '''
    x = torch.rand(2,3,4,4)
    m = torch.nn.Sequential(torch.nn.Conv2d(3,4,2),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(4,4,3,padding=1),
                            torch.nn.ReLU(),
                            Flatten(),
                            torch.nn.Linear(4*3*3,4))
    GC = GradCAM(m)
    maps = GC(x,submodule=None) # test using inputs as activations

    maps1 = GC(x[0].unsqueeze(0),submodule=m._modules['2']) # test using direct submodule hooks

    plt.subplot(1,2,1)
    plt.imshow(np.array(x[0].permute(1,2,0))) # plot input exapmle
    plt.subplot(1,2,2)
    plt.imshow(np.array(maps[0][0].detach())) # plot map example
    plt.show()
