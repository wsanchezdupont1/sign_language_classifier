"""

make_valset.py

Make the validation set by moving some images from /train to /val.

"""
import numpy as np
import os

def make_validation_set(traindir="/home/wjsd/Desktop/Coding/sign_language_classifier/asl_alphabet/train",
                        valdir="/home/wjsd/Desktop/Coding/sign_language_classifier/asl_alphabet/val",
                        val_frac=0.15):
    """
    make_validation_set

    Make the validation set by moving some images from traindir to valdir.
    Transfers np.floor(3000*val_frac) samples from each class into valdir. Uniformly samples images
    from class folders.

    inputs:
        traindir - (str) directory containing folders of train examples for the
                   the asl_dataset (traindir contains directories of class images)
        valdir - (str) directory to move validation images to
        val_frac - (float) fraction of training images to move to val folder. Equal number of samples per class.
    """

    if len(os.listdir(valdir)) == 0: # TODO: find a better criterion?
        make_empty_letterdirs(traindir=traindir,valdir=valdir)

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

def make_empty_letterdirs(traindir="/home/wjsd/Desktop/Coding/sign_language_classifier/train",
                          valdir="/home/wjsd/Desktop/Coding/sign_language_classifier/val"):
    """
    make_empty_letterdirs

    A function to create blank folders before transferring from train to val (in order to avoid "no such directory" error).
    Made specifically to be used with make_validation_set function.

    inputs:
        traindir - training data directory
        valdir - validation data directory
    """
    letters = os.listdir(traindir)
    for letter in letters:
        os.mkdir(os.path.join(valdir,letter))

def merge_trainset_valset(traindir="/home/wjsd/Desktop/Coding/sign_language_classifier/train",
                          valdir="/home/wjsd/Desktop/Coding/sign_language_classifier/val"):
    """
    merge_trainset_valset

    A function to put validation samples back into the training directories.

    inputs:
        traindir - training data directory
        valdir - validation data directory
    """
    # make sure directories exist
    if not os.path.exist(traindir):
        raise Exception("traindir does not exist!")
    elif not os.path.exist(valdir):
        raise Exception("valdir does not exist!")

    # define and sort letter directories
    trainletters = os.listdir(traindir)
    trainletters.sort()
    valletters = os.listdir(valdir)
    valletters.sort()

    # make sure they have the same letters
    for i in range(len(trainletters)):
        if trainletters[i] != valletters[i]:
            raise Exception("Letter directory difference between traindir and valdir!")

    # move the samples
    for letter in valletters:
        sampleNames = os.listdir(os.path.join(valdir,letter))
        for name in sampleNames:
            src = os.path.join(valdir,letter,name)
            dest = os.path.join(traindir,letter,name)
            os.rename(src,dest)

    for letter in os.listdir(traindir):
        if len(os.listdir(os.path.join(traindir,letter))) != 3000:
            raise Exception("Directory length wrong!")


# Run as script
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--traindir",type=str,default="/home/wjsd/Desktop/Coding/sign_language_classifier/train",help="Directory containing training data folders")
    parser.add_argument("--valdir",type=str,default="/home/wjsd/Desktop/Coding/sign_language_classifier/val",help="Directory containing validation data folders")
    parser.add_argument("--undo",action="store_true",help="Undo validation split by moving all validation samples back into the train directory")
    opts = parser.parse_args()

    if not opts.undo:
        make_validation_set(traindir=opts.traindir, valdir=opts.valdir) # create val set
    else:
        raise Exception("Please implement merge_trainset_valset")
        merge_trainset_valset(traindir=opts.traindir, valdir=opts.valdir)
