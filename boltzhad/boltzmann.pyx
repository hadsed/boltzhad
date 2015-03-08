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


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef isingenergy(np.float_t [:] z, 
                  np.float_t [:, :] J):
    """ Return the energy of the Hopfield network. """
    return -0.5*np.inner(z, np.inner(J, z))

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
cdef logit(x):
    return 1.0/(1.0 + np.exp(-x))

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
cdef inline contr_div(np.ndarray[np.float_t, ndim=1] state,
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
    gvbias = state[:nvisible]
    # calculate hidden unit state given the visibles (training vec) and
    # sample for the actual state using activation probabilities
    state[nvisible:] = (logit(np.dot(W.T, state[:nvisible]) + hbias) > 
                        np.random.rand(nhidden))
    # store initial state of visible units for bias update
    ghbias = state[nvisible:]
    # positive contribution
    grad = np.outer(state[:nvisible], state[nvisible:])
    # negative phase
    # loop over up-down passes
    for k in xrange(cdk):
        # resample visible units
        visreconprobs = logit(np.dot(W, state[nvisible:]) + vbias)
        state[:nvisible] = visreconprobs > np.random.rand(nvisible)
        # resample hidden units
        state[nvisible:] = (logit(np.dot(W.T, visreconprobs)) + hbias > 
                            np.random.rand(1, nhidden))
    # negative contribution
    grad -= np.outer(state[:nvisible], state[nvisible:])
    gvbias -= state[:nvisible]
    ghbias -= state[nvisible:]
    return grad, gvbias, ghbias


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
def train_restricted(np.float_t [:, :] data, 
                     np.ndarray[np.float_t, ndim=2] W,
                     np.ndarray[np.float_t, ndim=1] vbias,
                     np.ndarray[np.float_t, ndim=1] hbias,
                     float eta, 
                     int epochs, 
                     int cdk,
                     rng):
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
    @cdk is the number of contrastive divergence steps to
    take.

    @data should be in a 2D matrix where columns represent 
    training vectors and rows represent component variables.

    This version takes in a scipy.sparse matrix for the 
    weights @W and enforces this sparsity at each weight 
    update.
    """
    cdef int nvisible = W.shape[0]
    cdef int nhidden = W.shape[1]
    cdef int ep = 0
    cdef int idat = 0
    cdef np.ndarray[np.float_t, ndim=1] state = np.zeros(nvisible+nhidden)
    # check that the data vectors are the right length
    if W.shape[0] != data.shape[0]:
        print("Warning: data and weight matrix shapes don't match.")
    # training epochs
    for ep in xrange(epochs):
        # train W using MCMC for each training vector
        for idat in rng.permutation(range(data.shape[1])):
            # initialize state to the training vector
            state[:nvisible] = data[:,idat]
            # get gradient updates (weights, visible bias, hidden bias) from CD
            gw, gv, gh = contr_div(state, W, vbias, hbias, cdk)
            # update weights and biases
            W += gw*eta
            vbias += gv*eta
            hbias += gh*eta
    return W, vbias, hbias

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
def sample_restricted(np.ndarray[np.float_t, ndim=1] state,
                      np.ndarray[np.float_t, ndim=2] W,
                      np.ndarray[np.float_t, ndim=1] vbias,
                      np.ndarray[np.float_t, ndim=1] hbias,
                      int ksteps):
    """
    This method implements a @k-step Gibbs sampler for
    the hidden units of a restricted Boltzmann machine
    given some starting vector @start. @W is the coupling 
    matrix.
    """
    cdef int nvisible = W.shape[0]
    cdef int nhidden = W.shape[1]
    # sample hidden units given some starting state for visibles
    state[nvisible:] = (logit(np.dot(W.T, state[:nvisible]) + hbias) > 
                        np.random.rand(nhidden))
    # loop over up-down passes
    for k in xrange(ksteps):
        # sample visible units
        # visreconprobs = logit(np.dot(W, state[nvisible:]) + vbias)
        # state[:nvisible] = visreconprobs > np.random.rand(nvisible)
        state[:nvisible] = (logit(np.dot(W, state[nvisible:]) + vbias) > 
                            np.random.rand(nvisible))
        # sample hidden units
        state[nvisible:] = (logit(np.dot(W.T, state[:nvisible])) + hbias > 
                            np.random.rand(1, nhidden))
    return state
