'''

File: hopfield_digits.py
Author: Hadayat Seddiqi
Date: 02.15.15
Description: Try to train a Hopfield net on the 
             binary alphadigits dataset.

'''

import numpy as np
import scipy.sparse as sps
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
    return 1.0/(1.0 + np.exp(-x))

# number of visible units (must be equal to data length)
nvisible = 320
# number of hidden units
# nhidden = 50
# for plotting
nhidrow = 5  # essentially unbounded
nhidcol = 16  # should be less than 16
nhidden = nhidrow*nhidcol
# number of MCMC steps in CD
cdk = 20
# size of the minibatch for each gradient update
batchsize = 10
# learning rate
eta = 0.01
# training epochs
# (if we're low on data, we can set this higher)
epochs = 100
# Random number generator
seed = None
rng = np.random.RandomState(seed)

# number of sample inputs for testing
samples = 10
# fix up the data we want
classes = [10]#,11,12]
# max training vectors is 39
nperclass = 29
# up-down iterations for sampling trained network
kupdown = 100

# training data
datamat = sio.loadmat('data/binaryalphadigs.mat')
# construct list of training vecs for each class
datasp = np.asarray(
    [ [ datamat['dat'][j][k].flatten() 
        for k in xrange(nperclass) ]
      for j in classes ],
    dtype=np.float
)
# reshape to make 2D data matrix (neurons x training vecs)
datasp = datasp.reshape(-1, datasp.shape[-1]).T
# weight matrix (tiny random numbers)
scale = 1e-3
W = rng.rand(nvisible,nhidden)*scale
vbias = rng.rand(nvisible, 1)*scale
hbias = rng.rand(nhidden, 1)*scale
# train the weights (inplace)
print("Training...")
boltz.train_restricted(datasp, W, vbias, hbias, eta, epochs, 
                       cdk, batchsize, rng)
print("Training complete.")
# plot stuff
fig, ax = plt.subplots(1+2*len(classes),1)
border = 0
# create a matrix to hold all the stuff we want to plot
trainmat = -1.0*np.ones((20*len(classes), nperclass*16))
inpmat = -1.0*np.ones((20+nhidrow+1, samples*(16+border)))
outmat = -1.0*np.ones((20+nhidrow+1, samples*(16+border)))
classmats = {}
# first row is the training images
for itvec, tvec in enumerate(datasp.T):
    # how many training per class
    cpart = datasp.shape[1]/len(classes)
    # class id
    cidx = (itvec - (itvec % cpart)) / cpart
    # column index
    colidx = (itvec % cpart)*16
    # fill array
    trainmat[20*cidx:20*(cidx+1),colidx:colidx+16] = tvec.reshape(20,16).astype(int)
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
        state[:nvisible] = datamat['dat'][cls][inpidx].flatten()
        # calculate hidden unit probabilties given the visibles
        state[nvisible:] = (logit(np.dot(W.T, state[:nvisible])) > 
                            np.random.rand(nhidden))
        # input visible units
        colidx = isample*16
        colidxh = colidx
        classmats[cls]['inp'][:20,colidx:colidx+16] = \
                                state[:nvisible].reshape(20,16).astype(int)
        # input hidden units
        classmats[cls]['inp'][21:21+nhidrow,colidxh:colidxh+nhidcol] = \
                state[nvisible:].reshape(nhidrow,nhidcol).astype(int)
        # do some up-down samples
        print("Sampling for input #"+str(isample))
        state = boltz.sample_restricted(state, W, vbias.ravel(), hbias.ravel(), kupdown)
        print("Sampling #"+str(isample)+" done.")
        # output visibles
        classmats[cls]['out'][:20,colidx:colidx+16] = \
                                state[:nvisible].reshape(20,16).astype(int)
        # output hiddens
        classmats[cls]['out'][21:21+nhidrow,colidxh:colidxh+nhidcol] = \
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
