'''

File: boltz_mnist_sample.py
Author: Hadayat Seddiqi
Date: 03.22.15
Description: Sample trained restricted Boltzmann machine using CD.

'''

import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import cPickle as pk

import boltzhad.boltzmann as boltz
from boltzhad.utils import tile_raster_images

import matplotlib.animation as anim
import PIL.Image as Image


# def logit(x):
#     x[np.where(np.abs(x) < 30)[0]] = 1.0/(1.0 + np.exp(-x[np.abs(x) < 30]))
#     x[x > 30] = 1
#     x[x < -30] = 0
#     return x
def logit(x):
    """ Return output of the logistic function. """
    return 1.0/(1.0 + np.exp(-x))

classes = range(10)
nperclass = 1
kupdown = 100
nhidrow, nhidcol = 8, 28
# sample a test image
d = sio.loadmat('figs_boltz_mnist/model.mat')
W = d['w']
# vbias = np.zeros((784, 1))
# hbias = np.zeros((nhidrow*nhidcol, 1))
vbias = d['v']
hbias = d['h']
# training data
datamat = sio.loadmat('data/mnist_all.mat')
# construct list of training vecs for each class
dtest = np.vstack([ datamat['test'+str(d)][:nperclass]
                     for d in classes ]).astype(np.float)
# normalize
dtest /= float(255)
# train, valid, test = pk.load(open('mnist.pkl','rb'))
# data = valid
# idxsort = np.argsort(data[1])
pstates = []
state = np.zeros((784+nhidrow*nhidcol, 1))
fig, ax = plt.subplots(2, 2)
# remove annoying axes
for rax in ax:
    for kax in rax:
        kax.get_xaxis().set_visible(False)
        kax.get_yaxis().set_visible(False)
# loop over inputs we're going to start with
for itr in xrange(nperclass*len(classes)):
    # get test vector
    state[:784] = dtest[itr].reshape(784, 1)
    state[784:] = logit(np.dot(W.T, state[:784]) + hbias)
    # plot
    pvinp = ax[0,0].matshow(1-state[:784].reshape(28,28), cmap='Greys')
    phinp = ax[0,1].matshow(1-state[784:].reshape(nhidrow, nhidcol), cmap='Greys')
    # pvis = ax[1,0].matshow(1-state[:784].reshape(28,28), cmap='Greys')
    # phid = ax[1,1].matshow(1-state[784:].reshape(nhidrow, nhidcol), cmap='Greys')
    # animated plot
    for k in xrange(kupdown):
        hact = logit(np.dot(W.T, state[:784]) + hbias)
        state[784:] = hact > np.random.rand(nhidrow*nhidcol, 1)
        vact = logit(np.dot(W, state[784:]) + vbias)
        state[:784] = vact > np.random.rand(784, 1)
        ptitle = ax[1,1].text(0.5, 1.1, "%i Gibbs samples" % k,
                              horizontalalignment='center',
                              verticalalignment='bottom',
                              fontsize=16,
                              fontweight='bold',
                              transform=ax[1,1].transAxes)
        # pvis = ax[1,0].matshow(1-state[:784].reshape(28,28), cmap='Greys')
        # phid = ax[1,1].matshow(1-state[784:].reshape(nhidrow, nhidcol), cmap='Greys')
        pvis = ax[1,0].matshow(1-vact.reshape(28,28), cmap='Greys')
        phid = ax[1,1].matshow(1-hact.reshape(nhidrow, nhidcol), cmap='Greys')
        pstates.append([ptitle, pvis, phid, pvinp, phinp])
ani = anim.ArtistAnimation(fig, pstates, interval=50, blit=False)
print("Saving...")
ani.save('figs_boltz_mnist/samples_animation.gif', writer='imagemagick', fps=30)
print("Saving done.")
plt.show()
