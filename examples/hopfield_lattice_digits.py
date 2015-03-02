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

# Random number generator
seed = None
rng = np.random.RandomState(seed)
# training data
datamat = sio.loadmat('data/binaryalphadigs.mat')
# print datamat['classlabels']
# print datamat['classlabels'].size
# print datamat['dat'].shape
# num of neurons
neurons = 320 #datasp[:,0].size
classes = [10]#36
nperclass = 1
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
W = hopfield.rand_kblock_2d_lattice(20, 16, rng, k=320)
print W.nnz
# number of MCMC steps
nsteps = 10
# learning rate
eta = 0.01
# temperature
temp = 1.0
# training epochs
epochs = 100
# train the weights
W = W.todense()
W = hopfield.train_sparse(datasp, W, nsteps, eta, temp, epochs, rng)
# Print out data
# print("Training data (in order given to model):")
# for dvec in datasp.T:
#     print(np.asarray(dvec.reshape(20,16), dtype=int))
# Print out energies from trained model
print("Lowest states and their energies (sampled with simulated annealing):")
samples = 10
partitioning = (len(classes), samples)
sqrtsamples = int(np.sqrt(samples))
distr = np.zeros((neurons, samples*len(classes)))
# annealing schedule (lower temp makes it essentially like SGD)
asched = np.linspace(1.0, 0.9, 100)
# Generate list of nearest-neighbors for each spin
# neighbors = sa.GenerateNeighbors(neurons, sps.dok_matrix(W), 2*neurons)
print("Generating annealing samples from the trained model.")
# gather samples and plot them out
fig, axarr = plt.subplots(2*len(classes), samples+1)
for iclass, dclass in enumerate(classes):
    for isample in xrange(samples+1):
        # choose input vector (not from training data)
        trainidx = nperclass+isample-1
        state = np.asarray(
            bits2spins(datamat['dat'][dclass][trainidx].flatten()),
            dtype=np.float)
        # input row
        axarr[2*iclass,isample].imshow(state.reshape(20,16).astype(int),
                                     cmap='Greys', interpolation='nearest')
        axarr[2*iclass,isample].get_xaxis().set_visible(False)
        axarr[2*iclass,isample].get_yaxis().set_visible(False)
        if isample == 0:
            axarr[2*iclass,isample].set_title("Training")
        # anneal
        # sa.Anneal(asched, 1, state, neighbors, rng)
        sa.Anneal_dense(asched, 1, state, W, rng)
        # sa.Anneal_dense_parallel(asched, 1, state, W, 4)
        # recall row
        axarr[2*iclass+1,isample].imshow(state.reshape(20,16).astype(int),
                                       cmap='Greys', interpolation='nearest')
        axarr[2*iclass+1,isample].get_xaxis().set_visible(False)
        axarr[2*iclass+1,isample].get_yaxis().set_visible(False)
fig.subplots_adjust(left=0.01, bottom=0.0, right=0.99, 
                    top=0.90, wspace=0.001, hspace=0.001)
plt.tight_layout()
plt.show()
