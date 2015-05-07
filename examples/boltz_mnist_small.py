'''

File: boltz_mnist.py
Author: Hadayat Seddiqi
Date: 05.06.15
Description: Train a restricted Boltzmann machine
             on some small (14x14) MNIST data.

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
nvisible = 196
# number of hidden units
# nhidden = 50
# for plotting
nhidrow = 4  # essentially unbounded
nhidcol = 28  # should be less than 16
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
nperclass = 50
# up-down iterations for sampling trained network
kupdown_inp = 10
# use a persistent chain?
persistent = False
# update gradients with probabilities (not sampled states)?
useprobs = False

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
boltz.train_restricted(datasp, W, vbias, hbias, eta, wdecay,
                       epochs, cdk, batchsize, rng, debug,
                       persistent, useprobs)
print("Training complete.")
modelfname = 'figs_boltz_mnist/model.mat'
sio.savemat(modelfname, {'w': W, 'v': vbias, 'h': hbias})
params = {'cdk': cdk,
          'batchsize': batchsize,
          'learningrate': eta,
          'weightdecay': wdecay,
          'classes': classes,
          'trainingperclass': nperclass,
          'persistentchain': persistent,
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

# # plot training images
# trainmat = -1.0*np.ones((28*len(classes), nperclass*28))
# for itvec, tvec in enumerate(datasp.T):
#     # how many training per class
#     cpart = datasp.shape[1]/len(classes)
#     # class id
#     cidx = (itvec - (itvec % cpart)) / cpart
#     # column index
#     colidx = (itvec % cpart)*28
#     # fill array
#     trainmat[28*cidx:28*(cidx+1),colidx:colidx+28] = tvec.reshape(28,28).astype(int)
# # switch zeros and negative ones
# trainmat[trainmat == 0] = -2
# trainmat[trainmat == -1] = 0
# trainmat[trainmat == -2] = -1

# we'll need this
state = np.asarray(np.random.binomial(1, 0.5, nvisible+nhidden), dtype=np.float)
# loop through the classes
inpmat = -1.0*np.ones((28+nhidrow+1, samples*28))
outmat = -1.0*np.ones((28+nhidrow+1, samples*28))
classmats = {}
for icls, cls in enumerate(classes):
    classmats[cls] = { 'inp': inpmat.copy(), 
                       'out': outmat.copy() }
    # gather samples and add them to plotting matrix
    for isample in xrange(samples):
        colidx = isample*28
        colidxh = colidx
        # choose input vector (not from training data)
        inpidx = nperclass+isample
        # set visible units to a (normalized) test image
        state[:nvisible] = datamat['train'+str(cls)][inpidx]/float(255)
        # calculate hidden unit probabilties given the visibles
        state[nvisible:] = (logit(np.dot(W.T, state[:nvisible])) > 
                            np.random.rand(nhidden))
        # input visible units
        classmats[cls]['inp'][:28,colidx:colidx+28] = \
                                state[:nvisible].reshape(28,28).astype(int)
        # input hidden units
        classmats[cls]['inp'][29:29+nhidrow,colidxh:colidxh+nhidcol] = \
                state[nvisible:].reshape(nhidrow,nhidcol).astype(int)
        # do some up-down samples
        print("Sampling for input #"+str(isample))
        state = boltz.sample_restricted(state.reshape(state.size,1), 
                                        W, vbias, hbias, kupdown_inp).ravel()
        print("Sampling #"+str(isample)+" done.")
        # output visibles
        classmats[cls]['out'][:28,colidx:colidx+28] = \
                                state[:nvisible].reshape(28,28).astype(int)
        # output hiddens
        classmats[cls]['out'][29:29+nhidrow,colidxh:colidxh+nhidcol] = \
                state[nvisible:].reshape(nhidrow,nhidcol).astype(int)
# switch zeros and negative ones for a nicer picture
for k in classmats.iterkeys():
    classmats[k]['inp'][classmats[k]['inp'] == 0] = -2
    classmats[k]['inp'][classmats[k]['inp'] == -1] = 0
    classmats[k]['inp'][classmats[k]['inp'] == -2] = -1
    classmats[k]['out'][classmats[k]['out'] == 0] = -2
    classmats[k]['out'][classmats[k]['out'] == -1] = 0
    classmats[k]['out'][classmats[k]['out'] == -2] = -1
# plot inputs and reconstructions
ckeys = list(classmats.itervalues())
cols = 2 if len(ckeys) > 5 else 1
rows = len(classes) if len(ckeys) > 5 else 2*len(classes)
fig, ax = plt.subplots(rows, cols, figsize=(10,5))
fig.subplots_adjust(hspace=0, wspace=0.1, left=0.1, right=0.9, bottom=0.1, top=0.9)
iax = 0
# if less than 5 classes
if cols == 1:
    for val in ckeys:
        ax[iax].matshow(val['inp'], cmap=matplotlib.cm.binary)
        ax[iax+1].matshow(val['out'], cmap=matplotlib.cm.binary)
        iax += 2
# more than 5 classes (2 columns)
else:
    for val in ckeys[:len(ckeys)/2]:
        ax[iax,0].matshow(val['inp'], cmap=matplotlib.cm.binary)
        ax[iax+1,0].matshow(val['out'], cmap=matplotlib.cm.binary)
        iax += 2
    iax = 0
    for val in ckeys[len(ckeys)/2:]:
        ax[iax,1].matshow(val['inp'], cmap=matplotlib.cm.binary)
        ax[iax+1,1].matshow(val['out'], cmap=matplotlib.cm.binary)
        iax += 2
# remove axes
if isinstance(ax, list) or isinstance(ax, np.ndarray):
    for r in ax:
        if isinstance(r, list) or isinstance(r, np.ndarray):
            for kax in r:
                kax.get_xaxis().set_visible(False)
                kax.get_yaxis().set_visible(False)
        else:
            r.get_xaxis().set_visible(False)
            r.get_yaxis().set_visible(False)
else:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# save and get out
fig.savefig('figs_boltz_mnist/final.png', dpi=150)
plt.close(fig)
