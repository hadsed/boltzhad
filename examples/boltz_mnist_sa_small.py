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
import PIL.Image as Image
import json

import boltzhad.boltzmann as boltz
import boltzhad.sa as sa
from boltzhad.utils import tile_raster_images


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
nvisible = 14*14
# number of hidden units
# nhidden = 50
# for plotting
nhidrow = 4
nhidcol = 28
nhidden = nhidrow*nhidcol
# number of MCMC steps in CD
cdk = 5
# number of training examples in batch update
batchsize = 1
# learning rate
eta = 0.1
# training epochs
# (if we're low on data, we can set this higher)
epochs = 15
# rate of weight decay
wdecay = 0.0
# Random number generator
seed = None
rng = np.random.RandomState(seed)
# debug with output plots
debug = 1
# number of sample inputs for testing
samples = 20
# fix up the data we want
classes = range(10)
# max training vectors is (by index):
# [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
# min of that is 5421
nperclass = 500
# up-down iterations for sampling trained network
kupdown_inp = 10
# use a persistent chain?
persistent = False
# update gradients with probabilities (not sampled states)?
useprobs = True

# training data
datamat = sio.loadmat('data/mnist_small.mat')
# construct list of training vecs for each class
datasp = np.vstack([ datamat['train'+str(d)][:nperclass]
                     for d in classes ]).astype(np.float).T
# normalize
datasp /= float(255)
# weight matrix (tiny random numbers)
scale = 1.0
low = -4 * np.sqrt(6. / (nhidden + nvisible))*scale
high = 4 * np.sqrt(6. / (nhidden + nvisible))*scale
W = rng.uniform(size=(nvisible,nhidden), low=low, high=high)
vbias = rng.uniform(size=(nvisible,1), low=low, high=high)
hbias = rng.uniform(size=(nhidden,1), low=low, high=high)
# pvis = np.average([ datasp[:nvisible][:,k:k+nperclass]
#                     for k in xrange(0, len(classes)*nperclass, nperclass) ],
#                   axis=2).T
# vbias = pvis/(1.0-pvis)
# vbias[vbias > 0] = np.log(vbias[vbias > 0])
# vbias = np.zeros((nvisible,1))
# hbias = -4.0*np.ones((nhidden,1))*0.0
# train the weights (inplace)
print("Beginning training...")
sweeps = 1
tempstart = 10.0
tempend = 0.01
tannealingsteps = 10
trexp = (tempend/tempstart)**(1./tannealingsteps)
sched = np.array([ tempstart*trexp**k
                   for k in xrange(tannealingsteps) ])
useprobs = 0
boltz.train_restricted_sa(datasp, W, vbias, hbias, eta, wdecay,
                          epochs, sched, sweeps, rng, debug,
                          useprobs)
print("Training complete.")
modelfname = 'figs_boltz_mnist/model.mat'
sio.savemat(modelfname, {'w': W, 'v': vbias, 'h': hbias})
params = {'sched': list(sched),
          'sweeps': sweeps,
          'batchsize': batchsize,
          'learningrate': eta,
          'weightdecay': wdecay,
          'classes': classes,
          'trainingperclass': nperclass,
          'useprobs': useprobs,
          'initweights': (scale*low, scale*high)}
with open('figs_boltz_mnist/params.txt', 'w') as f:
    json.dump(params, f)
print("Model saved to file \"%s\"." % modelfname)
# plot the filters
image = Image.fromarray(
    tile_raster_images(
        X=W.T,
        img_shape=(np.sqrt(nvisible).astype(int),
                   np.sqrt(nvisible).astype(int)),
        tile_shape=(nhidrow, nhidcol),
        tile_spacing=(1, 1)
        )
    )
image.save('figs_boltz_mnist/filters.png')
print("Filters plotted.")
