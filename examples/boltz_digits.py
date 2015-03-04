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

# number of visible units (must be equal to data length)
nvisible = 320
# number of hidden units
nhidden = 100
# number of MCMC steps
nsteps = 10
# learning rate
eta = 0.1
# temperature
temp = 1.0
# training epochs
epochs = 100
# Random number generator
seed = None
rng = np.random.RandomState(seed)

# number of sample inputs for testing
samples = 10
# fix up the data we want
classes = [10]
nperclass = 10

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
W = rng.rand(nvisible,nhidden)*1e-1
W1 = W.copy()
# train the weights (change W inplace)
boltz.train_restricted(datasp, W, eta, epochs, 3, rng)
# print ''
# print("W has been trained.", (W1 - W))
# print ''
# os.lol

# Print out data
print("Training data (in order given to model):")
for dvec in datasp.T:
    print(np.asarray(dvec.reshape(20,16), dtype=int))
# Print out energies from trained model
print("Lowest states and their energies (sampled with simulated annealing):")
# annealing schedule (lower temp makes it essentially like SGD)
asched = np.linspace(2.0, 0.001, 100)
# create true J coupling matrix corresponding to W
J = sps.dok_matrix((nvisible+nhidden,nvisible+nhidden))
for row in xrange(nvisible):
    for col in xrange(nhidden):
        # offset by the right amount
        J[row,col+nvisible] = W[row,col]
# Generate list of nearest-neighbors for each spin
neighbors = sa.GenerateNeighbors(nvisible+nhidden, J, nvisible+1)
print("Generating annealing samples from the trained model.")
# gather samples and plot them out
fig, axarr = plt.subplots(4*len(classes), samples+1)
for iclass, dclass in enumerate(classes):
    for isample in xrange(samples+1):
        # choose input vector (not from training data)
        trainidx = nperclass+isample-1
        # initialize a random state
        state = np.asarray(np.random.binomial(1, 0.5, nvisible+nhidden),
                           dtype=np.float)
        # set visible units to a test image
        state[:nvisible] = np.asarray(
            bits2spins(datamat['dat'][dclass][trainidx].flatten()),
            dtype=np.float)
        # input row - visible units
        if isample == 0:
            axarr[2*iclass,isample].set_title("Inputs")
        axarr[2*iclass,isample].imshow(state[:nvisible].reshape(20,16).astype(int),
                                     cmap='Greys', interpolation='nearest')
        axarr[2*iclass,isample].get_xaxis().set_visible(False)
        axarr[2*iclass,isample].get_yaxis().set_visible(False)
        # input row - hidden units
        axarr[2*iclass+1,isample].imshow(state[nvisible:].reshape(
            np.sqrt(nhidden),np.sqrt(nhidden)).astype(int),
                                       cmap='Greys', interpolation='nearest')
        axarr[2*iclass+1,isample].get_xaxis().set_visible(False)
        axarr[2*iclass+1,isample].get_yaxis().set_visible(False)
        # anneal
        sa.Anneal(asched, 1, state, neighbors, rng)
        # output row
        axarr[2*iclass+2,isample].imshow(state[:nvisible].reshape(20,16).astype(int),
                                       cmap='Greys', interpolation='nearest')
        axarr[2*iclass+2,isample].get_xaxis().set_visible(False)
        axarr[2*iclass+2,isample].get_yaxis().set_visible(False)
        # output row - hidden units
        axarr[2*iclass+3,isample].imshow(state[nvisible:].reshape(
            np.sqrt(nhidden),np.sqrt(nhidden)).astype(int),
                                       cmap='Greys', interpolation='nearest')
        axarr[2*iclass+3,isample].get_xaxis().set_visible(False)
        axarr[2*iclass+3,isample].get_yaxis().set_visible(False)
fig.subplots_adjust(left=0.01, bottom=0.0, right=0.99, 
                    top=0.90, wspace=0.001, hspace=0.001)
plt.tight_layout()
plt.show()
