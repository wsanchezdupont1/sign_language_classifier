"""

make_valset.py

Make the validation set by moving some images from ..\\asl_alphabet\\train to ..\\asl_alphabet\\val.

"""
import numpy as np
import os

def make_validation_set(traindir="C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\asl_alphabet\\train",
                        valdir="C:\\Users\\Willis\\Desktop\\Sign Language Classifier\\asl_alphabet\\val",
                        val_frac=0.15):
    """
    make_validation_set

    Make the validation set by moving some images from ..\\asl_alphabet\\train to ..\\asl_alphabet\\val.
    Transfers np.floor(3000*val_frac) samples from each class into valdir. Uniformly samples images
    from class folders.

    inputs:
        traindir - (str) directory containing folders of train examples for the
                   the asl_dataset (traindir contains directories of class images)
        valdir - (str) directory to move validation images to
        val_frac - (float) fraction of training images to move to val folder. Equal number of samples per class.
    """

    #
    # 29 class directories,3000 samples per directory
    #
    # Loop through the classes and transfer np.floor(3000*val_frac) random images
    # to valdir for each.

    valSamplesPerClass = int(np.floor(3000*val_frac)) # train dataset originally has 3000 samples per class
    print('valSamplesPerClass =',int(valSamplesPerClass))
    letters = os.listdir(traindir)
    letters.sort() # [ 'A', 'B', ...'space'...'Y', 'Z']
    print('letters:')
    [print(x) for x in letters]

    # TODO: make sure this works...not sure it does currently...
    for letter in letters:
        idx_list = np.random.choice(3000,valSamplesPerClass,replace=False) # random image indices
        imNames = os.listdir(os.path.join(traindir,letter))
        for i in range(valSamplesPerClass): # note: possible to os.rename a batch of images at once?
            src = os.path.join(traindir, letter, imNames[idx_list[i]])
            dest = os.path.join(valdir,letter,imNames[idx_list[i]])
            os.rename(src,dest) # transfer file

    return valdir

make_validation_set() # create val set
