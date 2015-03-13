'''

File: boltz_mnist.py
Author: Hadayat Seddiqi
Date: 03.09.15
Description: Train a restricted Boltzmann machine
             on some MNIST data.

'''

import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

import boltzhad.hopfield as hopfield
import boltzhad.boltzmann as boltz
import boltzhad.sa as sa


def bits2spins(vec):
    """ Convert a bitvector @vec to a spinvector. """
    return np.array([ -1 if k == 1 else 1 for k in vec ])

def spins2bits(vec):
    """ Convert a spinvector @vec to a bitvector. """
    return np.array([ 0 if k == 1 else 1 for k in vec ])

def bitstr2spins(vec):
    """ Take a bitstring and return a spinvector. """
    a = np.array([ int(k) for k in vec ])
    return bits2spins(a)

def spins2bitstr(vec):
    """ Take a bitstring and return a spinvector. """
    a = spins2bits(a)
    return reduce(lambda x,y: x+y, [ str(k) for k in bvec ])

def logit(x):
    x[np.where(np.abs(x) < 30)[0]] = 1.0/(1.0 + np.exp(-x[np.abs(x) < 30]))
    x[x > 30] = 1
    x[x < -30] = 0
    return x

# number of visible units (must be equal to data length)
nvisible = 784
# number of hidden units
# nhidden = 50
# for plotting
nhidrow = 10  # essentially unbounded
nhidcol = 28  # should be less than 16
nhidden = nhidrow*nhidcol
# number of MCMC steps in CD
cdk = 15
# number of training examples in batch update
batchsize = 1
# learning rate
eta = 0.1
# training epochs
# (if we're low on data, we can set this higher)
epochs = 10
# rate of weight decay
wdecay = 0.001
# Random number generator
seed = None
rng = np.random.RandomState(seed)

# number of sample inputs for testing
samples = 20
# fix up the data we want
classes = [5]
# max training vectors is 6742
nperclass = 600
# up-down iterations for sampling trained network
kupdown = 10

# training data
datamat = sio.loadmat('data/mnist_all.mat')
# construct list of training vecs for each class
datasp = np.vstack([ datamat['train'+str(d)][:nperclass]
                     for d in classes ]).astype(np.float).T
# weight matrix (tiny random numbers)
scale = 1e-10
low = -4 * np.sqrt(6. / (nhidden + nvisible))*scale
high = 4 * np.sqrt(6. / (nhidden + nvisible))*scale
W = rng.uniform(size=(nvisible,nhidden), low=low, high=high)
# vbias = rng.uniform(size=(nvisible,1), low=low, high=high)
# hbias = rng.uniform(size=(nhidden,1), low=low, high=high)
# pvis = np.average([ datasp[:nvisible][:,k:k+nperclass]
#                     for k in xrange(0, len(classes)*nperclass, nperclass) ],
#                   axis=2).T
# vbias = pvis/(1.0-pvis)
# vbias[vbias > 0] = np.log(vbias[vbias > 0])
vbias = np.zeros((nvisible,1))
hbias = -4.0*np.ones((nhidden,1))*0.0
# train the weights (inplace)
print W
boltz.train_restricted(datasp, W, vbias, hbias, eta, wdecay,
                       epochs, cdk, batchsize, rng)
print W
# plot stuff
fig, ax = plt.subplots(1+2*len(classes),1)
border = 0
# create a matrix to hold all the stuff we want to plot
trainmat = -1.0*np.ones((28*len(classes), nperclass*28))
inpmat = -1.0*np.ones((28+nhidrow+1, samples*(28+border)))
outmat = -1.0*np.ones((28+nhidrow+1, samples*(28+border)))
classmats = {}
# first row is the training images
for itvec, tvec in enumerate(datasp.T):
    # how many training per class
    cpart = datasp.shape[1]/len(classes)
    # class id
    cidx = (itvec - (itvec % cpart)) / cpart
    # column index
    colidx = (itvec % cpart)*28
    # fill array
    trainmat[28*cidx:28*(cidx+1),colidx:colidx+28] = tvec.reshape(28,28).astype(int)
# loop through the classes
for icls, cls in enumerate(classes):
    classmats[cls] = { 'inp': inpmat.copy(), 
                       'out': outmat.copy() }
    # gather samples and add them to plotting matrix
    for isample in xrange(samples):
        # choose input vector (not from training data)
        inpidx = nperclass+isample-1
        # initialize a random state
        state = np.asarray(np.random.binomial(1, 0.5, nvisible+nhidden),
                           dtype=np.float)
        # set visible units to a test image
        state[:nvisible] = datamat['train'+str(cls)][inpidx]
        # calculate hidden unit probabilties given the visibles
        state[nvisible:] = (logit(np.dot(W.T, state[:nvisible])) > 
                            np.random.rand(nhidden))
        # input visible units
        colidx = isample*28
        colidxh = colidx
        classmats[cls]['inp'][:28,colidx:colidx+28] = \
                                state[:nvisible].reshape(28,28).astype(int)
        # input hidden units
        classmats[cls]['inp'][29:29+nhidrow,colidxh:colidxh+nhidcol] = \
                state[nvisible:].reshape(nhidrow,nhidcol).astype(int)
        # do some up-down samples
        state = boltz.sample_restricted(state.reshape(state.size, 1), W, 
                                        vbias, hbias, kupdown)
        # output visibles
        classmats[cls]['out'][:28,colidx:colidx+28] = \
                                state[:nvisible].reshape(28,28).astype(int)
        # output hiddens
        classmats[cls]['out'][29:29+nhidrow,colidxh:colidxh+nhidcol] = \
                state[nvisible:].reshape(nhidrow,nhidcol).astype(int)
# switch zeros and negative ones
trainmat[trainmat == 0] = -2
trainmat[trainmat == -1] = 0
trainmat[trainmat == -2] = -1
for k in classmats.iterkeys():
    classmats[k]['inp'][classmats[k]['inp'] == 0] = -2
    classmats[k]['inp'][classmats[k]['inp'] == -1] = 0
    classmats[k]['inp'][classmats[k]['inp'] == -2] = -1
    classmats[k]['out'][classmats[k]['out'] == 0] = -2
    classmats[k]['out'][classmats[k]['out'] == -1] = 0
    classmats[k]['out'][classmats[k]['out'] == -2] = -1
# plot them
ax[0].set_title("Training data")
ax[0].matshow(trainmat, cmap=matplotlib.cm.binary)
# input rows
iax = 1
for val in classmats.itervalues():
    ax[iax].set_title("Inputs (visible then hidden units)")
    ax[iax].matshow(val['inp'], cmap=matplotlib.cm.binary)
    ax[iax+1].set_title("Result (after "+str(kupdown)+" up-down samples)")
    ax[iax+1].matshow(val['out'], cmap=matplotlib.cm.binary)
    iax += 2
# remove axes
for kax in ax:
    kax.get_xaxis().set_visible(False)
    kax.get_yaxis().set_visible(False)
plt.show()
