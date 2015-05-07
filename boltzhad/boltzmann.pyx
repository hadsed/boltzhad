'''

File: boltzmann.pyx
Author: Hadayat Seddiqi
Date: 02.17.15
Description: Implement routines for training Boltzmann machines.

'''

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp as cexp
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX
import matplotlib.pyplot as plt
from utils import tile_raster_images
import PIL.Image as Image

import boltzhad.sa as sa

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef isingenergy(np.float_t [:] z, 
                  np.float_t [:, :] J):
    """ Return the energy of the Hopfield network. """
    return -0.5*np.inner(z, np.inner(J, z))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cdef inline np.ndarray[np.float_t, ndim=2] logitopt(np.ndarray[np.float_t, ndim=2] x):
    """ Return output of the logistic function. """
    cdef np.ndarray[np.float_t, ndim=2] y = x.copy()
    y[np.abs(y) < 30] = 1.0/(1.0 + np.exp(-y[np.abs(y) < 30]))
    y[y < -30] = 0
    y[y > 30] = 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cdef inline np.ndarray[np.float_t, ndim=2] logit(np.ndarray[np.float_t, ndim=2] x):
    """ Return output of the logistic function. """
    return 1.0/(1.0 + np.exp(-x))

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cdef inline contr_div(np.ndarray[np.float_t, ndim=2] state,
                      np.float_t [:,:] W,
                      np.float_t [:] vbias,
                      np.float_t [:] hbias,
                      int cdk):
    """
    Implement contrastive divergence with @cdk up-down
    steps using given weights @W and binary start state
    @state, whose first @nvisible elements give the
    visible unit states, and the rest give the hiddens.
    @W has shape (@nvisible, @nhidden).
    """
    cdef int nvisible = vbias.size
    cdef int nhidden = hbias.size
    cdef int k = 0
    cdef np.ndarray[np.float_t, ndim=2] grad = np.zeros((nvisible, nhidden))
    cdef np.ndarray[np.float_t, ndim=1] gvbias = np.empty(nvisible)
    cdef np.ndarray[np.float_t, ndim=1] ghbias = np.empty(nhidden)
    cdef np.ndarray[np.float_t, ndim=1] visreconprobs = np.empty(nvisible)
    # positive phase
    # store initial state of visible units for bias update
    gvbias = state[:nvisible, 0]
    # calculate hidden unit state given the visibles (training vec) and
    # sample for the actual state using activation probabilities
    state[nvisible:, 0] = (logit(np.dot(W.T, state[:nvisible, 0]) + hbias) > 
                           np.random.rand(nhidden))
    # store initial state of visible units for bias update
    ghbias = state[nvisible:, 0]
    # positive contribution
    grad = np.outer(state[:nvisible], state[nvisible:])
    # negative phase
    # loop over up-down passes
    for k in xrange(cdk):
        # resample visible units
        visreconprobs = logit(np.dot(W, state[nvisible:, 0]) + vbias)
        state[:nvisible, 0] = visreconprobs > np.random.rand(nvisible)
        # resample hidden units
        state[nvisible:, 0] = (logit(np.dot(W.T, visreconprobs)) + hbias > 
                               np.random.rand(1, nhidden))
    # negative contribution
    grad -= np.outer(state[:nvisible], state[nvisible:])
    gvbias -= state[:nvisible, 0]
    ghbias -= state[nvisible:, 0]
    return grad, gvbias, ghbias

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cdef inline contr_div_batch(np.ndarray[np.float_t, ndim=2] state,
                            np.ndarray[np.float_t, ndim=2] pchain,
                            np.float_t [:,:] W,
                            np.float_t [:,:] vbias,
                            np.float_t [:,:] hbias,
                            np.ndarray[np.float_t, ndim=2] grad,
                            np.ndarray[np.float_t, ndim=2] gvbias,
                            np.ndarray[np.float_t, ndim=2] ghbias,
                            int cdk,
                            int persistent,
                            int useprobs):
    """
    Implement contrastive divergence with @cdk up-down
    steps using given weights @W and binary start state
    @state, whose first @nvisible elements give the
    visible unit states, and the rest give the hiddens.
    @W has shape (@nvisible, @nhidden).

    Edit the above.

    This is the batch version, where @state has rows of
    length equal to the total units in the RBM, columns
    equal to the batchsize.
    """
    cdef int nvisible = vbias.shape[0]
    cdef int nhidden = hbias.shape[0]
    cdef int batchsize = state.shape[1]
    cdef int k = 0
    cdef np.ndarray[np.float_t, ndim=2] vreconprobs = np.empty((nvisible, batchsize))
    cdef np.ndarray[np.float_t, ndim=2] hreconprobs = np.empty((nhidden, batchsize))
    # positive phase
    if persistent:
        # initialize hidden units from persistent chain
        state[nvisible:] = pchain.copy()
    else:
        # calculate hidden unit state given the visibles (training vec)
        state[nvisible:] = logit(np.dot(W.T, state[:nvisible]) + hbias)
    if useprobs:
        # positive contribution (use the probabilities)
        grad[:] = np.dot(state[:nvisible], state[nvisible:].T)
        gvbias[:] = state[:nvisible].sum(axis=1).reshape(nvisible,1)
        ghbias[:] = state[nvisible:].sum(axis=1).reshape(nhidden,1)
        # sample for the actual state using activation probabilities
        state[nvisible:] = state[nvisible:] > np.random.rand(nhidden,batchsize)
    else:
        # sample for the actual state using activation probabilities
        state[nvisible:] = state[nvisible:] > np.random.rand(nhidden,batchsize)
        # positive contribution (use states)
        grad[:] = np.dot(state[:nvisible], state[nvisible:].T)
        gvbias[:] = state[:nvisible].sum(axis=1).reshape(nvisible,1)
        ghbias[:] = state[nvisible:].sum(axis=1).reshape(nhidden,1)
    # negative phase
    # loop over up-down passes (using the persistent negative Gibbs chain)
    for k in xrange(cdk):
        # resample visible units (must use hidden states, not probabilities)
        vreconprobs = logit(np.dot(W, state[nvisible:]) + vbias)
        state[:nvisible] = vreconprobs > np.random.rand(nvisible,batchsize)
        # resample hidden units
        hreconprobs = logit(np.dot(W.T, state[:nvisible]) + hbias)
        state[nvisible:] = hreconprobs > np.random.rand(nhidden,batchsize)
    # negative contribution
    if useprobs:
        grad[:] = (grad - np.dot(vreconprobs, hreconprobs.T))/float(batchsize)
        gvbias[:] = ((gvbias.T - vreconprobs.sum(axis=1)) /
                     float(batchsize)).T
        ghbias[:] = ((ghbias.T - hreconprobs.sum(axis=1)) /
                     float(batchsize)).T
    else:
        grad[:] = (grad - np.dot(state[:nvisible], state[nvisible:].T))/float(batchsize)
        gvbias[:] = ((gvbias.T - state[:nvisible].sum(axis=1)) /
                     float(batchsize)).T
        ghbias[:] = ((ghbias.T - state[nvisible:].sum(axis=1)) /
                     float(batchsize)).T
    if persistent:
        # save last state of hidden units in the persistent chain
        pchain[:] = state[nvisible:].copy()

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
def train_restricted(np.float_t [:, :] data, 
                     np.ndarray[np.float_t, ndim=2] W,
                     np.ndarray[np.float_t, ndim=2] vbias,
                     np.ndarray[np.float_t, ndim=2] hbias,
                     float eta, 
                     float wdecay,
                     int epochs,
                     int cdk,
                     int batchsize,
                     rng,
                     int debug,
                     int persistent,
                     int useprobs):
    """
    Train a restricted Boltzmann machine on @data. Number of 
    Monte Carlo steps in the Gibbs sampler is @nsteps, @T
    being the temperature, @eta the learning rate, and 
    @epochs the number of times we loop through the data 
    to train on (in random order each time). @nvisible and
    @nhidden denote the number of visible and hidden units,
    respectively. @W denotes the connections between them,
    and a given state vector will have the following labels:

    state[0:@nvisible] = data[:,k]
    state[@nvisible:@nhidden] = random_ints(0,1)

    where k is some random training vector index. So, the 
    first set of indices denote visibles, and the rest are
    hidden units. @W should have rank @nvisible+@nhidden.
    @cdk is the number of (persistent) contrastive divergence
    steps to take.

    @data should be in a 2D matrix where columns represent 
    training vectors and rows represent component variables.

    @batchsize should be 1 if we only want to do weight
    updates on single training instances, otherwise set to
    desired batchsize.
    """
    cdef int nvisible = W.shape[0]
    cdef int nhidden = W.shape[1]
    cdef int ep = 0
    cdef int idat = 0
    cdef int dstep = 0
    cdef np.float_t [:,:] firstdata = data[:,0:1000].copy()
    cdef np.ndarray[np.float_t, ndim=2] state = \
        np.empty((nvisible+nhidden, batchsize))
    cdef np.ndarray[np.float_t, ndim=2] pchain = \
        np.zeros((nhidden, batchsize)).astype(np.float)
    cdef np.ndarray[np.float_t, ndim=2] gw = np.zeros((nvisible, nhidden))
    cdef np.ndarray[np.float_t, ndim=2] gv = np.zeros((nvisible, 1))
    cdef np.ndarray[np.float_t, ndim=2] gh = np.zeros((nhidden, 1))
    cdef int nhidrow = nhidden/28
    cdef int nhidcol = 28
    cdef int row, col
    # check that the data vectors are the right length
    if W.shape[0] != data.shape[0]:
        print("Warning: data and weight matrix shapes don't match.")
    # check that we didn't have larger batchsize than actual datapoints
    if data.shape[1] < batchsize:
        print("Warning: batchsize larger than number of data points.")
    if debug:
        fig, ax = plt.subplots(1,1)#,figsize=(10,10))
        # examine hidden activations before training
        ax.matshow(logit(np.dot(W.T, firstdata) + hbias), 
                   cmap='Greys')
        plt.savefig('figs_boltz_mnist/hidden_act_0.png')
        # plot histograms of weights and biases, and of the last gradient
        figh, axh = plt.subplots(2,3,figsize=(20,10))
        axh[0,0].hist(vbias)
        axh[0,0].set_title("vbias mean = "+str(np.mean(np.fabs(vbias))))
        axh[0,1].hist(W.ravel())
        axh[0,1].set_title("W mean = "+str(np.mean(np.fabs(W.ravel()))))
        axh[0,2].hist(hbias)
        axh[0,2].set_title("hbias mean = "+str(np.mean(np.fabs(hbias))))
        axh[1,0].hist(gv*eta)
        axh[1,0].set_title("dvbias mean = "+str(np.mean(np.fabs(gv*eta))))
        axh[1,1].hist(gw.ravel()*eta)
        axh[1,1].set_title("dW mean = "+str(np.mean(np.fabs(gw.ravel()*eta))))
        axh[1,2].hist(gh*eta)
        axh[1,2].set_title("dhbias mean = "+str(np.mean(np.fabs(gh*eta))))
        figh.savefig('figs_boltz_mnist/hist_ep0.png')
        plt.close(figh)
        # for plotting filters
        figf, axf = plt.subplots(1, 1, figsize=(20,15))
        filtermat = np.zeros((28*nhidrow, 28*nhidcol))
        # training epochs
        for ep in xrange(epochs):
            print("\tEpoch %i" % ep)
            # randomize order of data to reduce bias
            data = rng.permutation(data.T).T
            # train W using MCMC for each training vector
            for idat in xrange(data.shape[1]/batchsize):
                # initialize state to the training vector
                dstep = min(idat+batchsize, data.shape[1])
                state[:nvisible] = data[:,idat:dstep]
                # calculate gradient updates (weights, visible bias, hidden bias)
                contr_div_batch(state, pchain, W, vbias, hbias, gw, gv, gh, 
                                cdk, persistent, useprobs)
                # update weights and biases
                W += (gw - W*wdecay)*eta
                vbias += gv*eta
                hbias += gh*eta
            # plot histograms of weights and biases, and of the last gradient
            figh, axh = plt.subplots(2,3,figsize=(20,10))
            axh[0,0].hist(vbias)
            axh[0,0].set_title("vbias mean = "+str(np.mean(np.fabs(vbias))))
            axh[0,1].hist(W.ravel())
            axh[0,1].set_title("W mean = "+str(np.mean(np.fabs(W.ravel()))))
            axh[0,2].hist(hbias)
            axh[0,2].set_title("hbias mean = "+str(np.mean(np.fabs(hbias))))
            axh[1,0].hist(gv*eta)
            axh[1,0].set_title("dvbias mean = "+str(np.mean(np.fabs(gv*eta))))
            axh[1,1].hist(gw.ravel()*eta)
            axh[1,1].set_title("dW mean = "+str(np.mean(np.fabs(gw.ravel()*eta))))
            axh[1,2].hist(gh*eta)
            axh[1,2].set_title("dhbias mean = "+str(np.mean(np.fabs(gh*eta))))
            figh.savefig('figs_boltz_mnist/hist_ep'+str(ep+1))#+'_d'+str(idat))
            plt.close(figh)
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
            image.save('figs_boltz_mnist/filters_%d.png' % (ep+1))
            # examine hidden activations during training
            ax.matshow(logit(np.dot(W.T, firstdata) + hbias), 
                       cmap='Greys')
            fig.savefig('figs_boltz_mnist/hidden_act_'+str(ep+1)+'.png')
        plt.close(figf)
        plt.close(fig)
    else:
        # training epochs
        for ep in xrange(epochs):
            # randomize order of data to reduce bias
            data = rng.permutation(data.T).T
            # train W using MCMC for each training vector
            for idat in xrange(data.shape[1]/batchsize):
                # initialize state to the training vector
                dstep = min(idat+batchsize, data.shape[1])
                state[:nvisible] = data[:,idat:dstep]
                # calculate gradient updates (weights, visible bias, hidden bias)
                contr_div_batch(state, pchain, W, vbias, hbias, gw, gv, gh, 
                                cdk, persistent, useprobs)
                # update weights and biases
                W += (gw - W*wdecay)*eta
                vbias += gv*eta
                hbias += gh*eta
    return W, vbias, hbias

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cdef inline sample_sa(np.ndarray[np.float_t, ndim=2] state,
                      np.float_t [:] sched,
                      int sweeps,
                      np.ndarray[np.float_t, ndim=2] J,
                      np.float_t [:,:] W,
                      np.float_t [:,:] vbias,
                      np.float_t [:,:] hbias,
                      np.ndarray[np.float_t, ndim=2] grad,
                      np.ndarray[np.float_t, ndim=2] gvbias,
                      np.ndarray[np.float_t, ndim=2] ghbias,
                      int useprobs,
                      rng):
    """
    """
    cdef int nvisible = vbias.shape[0]
    cdef int nhidden = hbias.shape[0]
    cdef int batchsize = 1
    cdef np.ndarray[np.float_t, ndim=1] stateitr = state[:,0].copy()
    cdef np.ndarray[np.float_t, ndim=1] stateavg = state[:,0].copy()
    # positive phase
    # calculate hidden unit state given the visibles (training vec)
    state[nvisible:] = logit(np.dot(W.T, state[:nvisible]) + hbias)
    if useprobs:
        # positive contribution (use the probabilities)
        grad[:] = np.dot(state[:nvisible], state[nvisible:].T)
        gvbias[:] = state[:nvisible].sum(axis=1).reshape(nvisible,1)
        ghbias[:] = state[nvisible:].sum(axis=1).reshape(nhidden,1)
        # sample for the actual state using activation probabilities
        state[nvisible:] = state[nvisible:] > np.random.rand(nhidden,batchsize)
    else:
        # sample for the actual state using activation probabilities
        state[nvisible:] = state[nvisible:] > np.random.rand(nhidden,batchsize)
        # positive contribution (use states)
        grad[:] = np.dot(state[:nvisible], state[nvisible:].T)
        gvbias[:] = state[:nvisible].sum(axis=1).reshape(nvisible,1)
        ghbias[:] = state[nvisible:].sum(axis=1).reshape(nhidden,1)
    # negative phase
    # run simulated annealing to estimate partition function
    # set diagonal to the biases
    J[np.diag_indices(nvisible+nhidden)] = np.concatenate((vbias[:,0], 
                                                           hbias[:,0]))
    # # loop over up-down passes (using the persistent negative Gibbs chain)
    # for k in xrange(cdk):
    #     # resample visible units (must use hidden states, not probabilities)
    #     vreconprobs = logit(np.dot(W, state[nvisible:]) + vbias)
    #     state[:nvisible] = vreconprobs > np.random.rand(nvisible,batchsize)
    #     # resample hidden units
    #     hreconprobs = logit(np.dot(W.T, state[:nvisible]) + hbias)
    #     state[nvisible:] = hreconprobs > np.random.rand(nhidden,batchsize)
    # negative contribution
    if useprobs:
        # take some samples
        for j in xrange(20):
            # convert state from bits to spins
            stateitr = sa.bits2spins(state[:,0].copy())
            # anneal
            sa.Anneal_dense(sched, sweeps, stateitr, J, rng)
            # accumulate
            stateavg += stateitr
        # switch back to binary representation (probability of unit k being on)
        state[:,0] = (stateavg/20.0 + 1)/2.0
        # update gradients
        grad[:] = ((grad - np.dot(state[:nvisible], state[nvisible:].T)) /
                   float(batchsize))
        gvbias[:] = ((gvbias.T - state[:nvisible].sum(axis=1)) /
                     float(batchsize)).T
        ghbias[:] = ((ghbias.T - state[nvisible:].sum(axis=1)) /
                     float(batchsize)).T
        # grad[:] = (grad - np.dot(vreconprobs, hreconprobs.T))/float(batchsize)
        # gvbias[:] = ((gvbias.T - vreconprobs.sum(axis=1)) /
        #              float(batchsize)).T
        # ghbias[:] = ((ghbias.T - hreconprobs.sum(axis=1)) /
        #              float(batchsize)).T
    else:
        # convert state from bits to spins
        # stateitr = sa.bits2spins(state[:,0].copy())
        # random initial spin-state
        stateitr = np.asarray(2*rng.random_integers(0,1,nvisible+nhidden) - 1,
                              dtype=np.float)
        # anneal
        sa.Anneal_dense(sched, sweeps, stateitr, J, rng)
        # convert back to binary and record
        state[:,0] = sa.spins2bits(stateitr)
        # print state[:,0]
        # update gradients
        grad[:] = ((grad - np.dot(state[:nvisible], state[nvisible:].T)) /
                   float(batchsize))
        gvbias[:] = ((gvbias.T - state[:nvisible].sum(axis=1)) /
                     float(batchsize)).T
        ghbias[:] = ((ghbias.T - state[nvisible:].sum(axis=1)) /
                     float(batchsize)).T

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
def train_restricted_sa(np.float_t [:, :] data, 
                        np.ndarray[np.float_t, ndim=2] W,
                        np.ndarray[np.float_t, ndim=2] vbias,
                        np.ndarray[np.float_t, ndim=2] hbias,
                        float eta, 
                        float wdecay,
                        int epochs,
                        np.ndarray[np.float_t, ndim=1] sched,
                        int sweeps,
                        rng,
                        int debug,
                        int useprobs):
    """
    Train a restricted Boltzmann machine on @data. Number of 
    Monte Carlo steps in the Gibbs sampler is @nsteps, @T
    being the temperature, @eta the learning rate, and 
    @epochs the number of times we loop through the data 
    to train on (in random order each time). @nvisible and
    @nhidden denote the number of visible and hidden units,
    respectively. @W denotes the connections between them,
    and a given state vector will have the following labels:

    state[0:@nvisible] = data[:,k]
    state[@nvisible:@nhidden] = random_ints(0,1)

    where k is some random training vector index. So, the 
    first set of indices denote visibles, and the rest are
    hidden units. @W should have rank @nvisible+@nhidden.
    @cdk is the number of (persistent) contrastive divergence
    steps to take.

    @data should be in a 2D matrix where columns represent 
    training vectors and rows represent component variables.

    @batchsize should be 1 if we only want to do weight
    updates on single training instances, otherwise set to
    desired batchsize.
    """
    cdef int nvisible = W.shape[0]
    cdef int nhidden = W.shape[1]
    cdef int ep = 0
    cdef int idat = 0
    cdef int dstep = 0
    cdef int batchsize = 1  # edit later!!!!
    cdef np.float_t [:,:] firstdata = data[:,0:1000].copy()
    cdef np.ndarray[np.float_t, ndim=2] state = \
        np.empty((nvisible+nhidden, batchsize))
    cdef np.ndarray[np.float_t, ndim=2] pchain = \
        np.zeros((nhidden, batchsize)).astype(np.float)
    cdef np.ndarray[np.float_t, ndim=2] J = np.zeros((nvisible+nhidden,
                                                      nvisible+nhidden))
    cdef np.ndarray[np.float_t, ndim=2] gw = np.zeros((nvisible, nhidden))
    cdef np.ndarray[np.float_t, ndim=2] gv = np.zeros((nvisible, 1))
    cdef np.ndarray[np.float_t, ndim=2] gh = np.zeros((nhidden, 1))
    cdef int nhidrow = nhidden/28
    cdef int nhidcol = 28
    cdef int row, col
    # check that the data vectors are the right length
    if W.shape[0] != data.shape[0]:
        print("Warning: data and weight matrix shapes don't match.")
    # check that we didn't have larger batchsize than actual datapoints
    if data.shape[1] < batchsize:
        print("Warning: batchsize larger than number of data points.")
    # setup the J matrix for SA
    J[:nvisible, nvisible:] = W.copy()
    # then make W a view onto J's dense elements
    W = J[:nvisible, nvisible:]

    if debug:
        fig, ax = plt.subplots(1,1)#,figsize=(10,10))
        # examine hidden activations before training
        ax.matshow(logit(np.dot(W.T, firstdata) + hbias), 
                   cmap='Greys')
        plt.savefig('figs_boltz_mnist_sa/hidden_act_0.png')
        # plot histograms of weights and biases, and of the last gradient
        figh, axh = plt.subplots(2,3,figsize=(20,10))
        axh[0,0].hist(vbias)
        axh[0,0].set_title("vbias mean = "+str(np.mean(np.fabs(vbias))))
        axh[0,1].hist(W.ravel())
        axh[0,1].set_title("W mean = "+str(np.mean(np.fabs(W.ravel()))))
        axh[0,2].hist(hbias)
        axh[0,2].set_title("hbias mean = "+str(np.mean(np.fabs(hbias))))
        axh[1,0].hist(gv*eta)
        axh[1,0].set_title("dvbias mean = "+str(np.mean(np.fabs(gv*eta))))
        axh[1,1].hist(gw.ravel()*eta)
        axh[1,1].set_title("dW mean = "+str(np.mean(np.fabs(gw.ravel()*eta))))
        axh[1,2].hist(gh*eta)
        axh[1,2].set_title("dhbias mean = "+str(np.mean(np.fabs(gh*eta))))
        figh.savefig('figs_boltz_mnist_sa/hist_ep0.png')
        plt.close(figh)
        # for plotting filters
        figf, axf = plt.subplots(1, 1, figsize=(20,15))
        filtermat = np.zeros((28*nhidrow, 28*nhidcol))
        # training epochs
        for ep in xrange(epochs):
            print("\tEpoch %i" % ep)
            # randomize order of data to reduce bias
            data = rng.permutation(data.T).T
            # train W using MCMC for each training vector
            for idat in xrange(data.shape[1]/batchsize):
                # initialize state to the training vector
                dstep = min(idat+batchsize, data.shape[1])
                state[:nvisible] = data[:,idat:dstep]
                # calculate gradient updates (weights, visible bias, hidden bias)
                sample_sa(state, sched, sweeps, J, W, vbias, hbias,
                          gw, gv, gh, useprobs, rng)
                # contr_div_batch(state, pchain, W, vbias, hbias, gw, gv, gh, 
                #                 cdk, persistent, useprobs)
                # update weights and biases
                # print gw.sum(axis=1), gv.T, gh.T
                W += (gw - W*wdecay)*eta
                vbias += gv*eta
                hbias += gh*eta
            # plot histograms of weights and biases, and of the last gradient
            figh, axh = plt.subplots(2,3,figsize=(20,10))
            axh[0,0].hist(vbias)
            axh[0,0].set_title("vbias mean = "+str(np.mean(np.fabs(vbias))))
            axh[0,1].hist(W.ravel())
            axh[0,1].set_title("W mean = "+str(np.mean(np.fabs(W.ravel()))))
            axh[0,2].hist(hbias)
            axh[0,2].set_title("hbias mean = "+str(np.mean(np.fabs(hbias))))
            axh[1,0].hist(gv*eta)
            axh[1,0].set_title("dvbias mean = "+str(np.mean(np.fabs(gv*eta))))
            axh[1,1].hist(gw.ravel()*eta)
            axh[1,1].set_title("dW mean = "+str(np.mean(np.fabs(gw.ravel()*eta))))
            axh[1,2].hist(gh*eta)
            axh[1,2].set_title("dhbias mean = "+str(np.mean(np.fabs(gh*eta))))
            figh.savefig('figs_boltz_mnist_sa/hist_ep'+str(ep+1))#+'_d'+str(idat))
            plt.close(figh)
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
            image.save('figs_boltz_mnist_sa/filters_%d.png' % (ep+1))
            # examine hidden activations during training
            ax.matshow(logit(np.dot(W.T, firstdata) + hbias), 
                       cmap='Greys')
            fig.savefig('figs_boltz_mnist_sa/hidden_act_'+str(ep+1)+'.png')
        plt.close(figf)
        plt.close(fig)
    else:
        # training epochs
        for ep in xrange(epochs):
            # randomize order of data to reduce bias
            data = rng.permutation(data.T).T
            # train W using MCMC for each training vector
            for idat in xrange(data.shape[1]/batchsize):
                # initialize state to the training vector
                dstep = min(idat+batchsize, data.shape[1])
                state[:nvisible] = data[:,idat:dstep]
                # calculate gradient updates (weights, visible bias, hidden bias)
                sample_sa(state, sched, sweeps, J, W, vbias, hbias, 
                          gw, gv, gh, useprobs, rng)
                # contr_div_batch(state, pchain, W, vbias, hbias, gw, gv, gh, 
                #                 cdk, persistent, useprobs)
                # update weights and biases
                W += (gw - W*wdecay)*eta
                vbias += gv*eta
                hbias += gh*eta
    return W, vbias, hbias

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
def sample_restricted(np.ndarray[np.float_t, ndim=2] state,
                      np.ndarray[np.float_t, ndim=2] W,
                      np.ndarray[np.float_t, ndim=2] vbias,
                      np.ndarray[np.float_t, ndim=2] hbias,
                      int ksteps):
    """
    This method implements a @k-step Gibbs sampler for
    the hidden units of a restricted Boltzmann machine
    given some starting vector @start. @W is the coupling 
    matrix.
    """
    cdef int nvisible = W.shape[0]
    cdef int nhidden = W.shape[1]
    cdef int batchsize = 1
    # sample hidden units given some starting state for visibles
    state[nvisible:] = (logit(np.dot(W.T, state[:nvisible]) + hbias) > 
                        np.random.rand(nhidden,batchsize))
    # loop over up-down passes
    for k in xrange(ksteps):
        # sample visible units
        state[:nvisible] = (logit(np.dot(W, state[nvisible:]) + vbias) > 
                            np.random.rand(nvisible,batchsize))
        # sample hidden units
        state[nvisible:] = (logit(np.dot(W.T, state[:nvisible])) + hbias > 
                            np.random.rand(nhidden, batchsize))
    return state
