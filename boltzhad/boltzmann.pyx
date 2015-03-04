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
cpdef contr_div(np.ndarray[np.float_t, ndim=1] state,
                np.float_t [:,:] W,
                int nvisible,
                int nhidden,
                int cdk):
    """
    Implement contrastive divergence with @cdk up-down
    steps using given weights @W and binary start state
    @state, whose first @nvisible elements give the
    visible unit states, and the rest give the hiddens.
    @W has shape (@nvisible, @nhidden).
    """

    # todo: take care of bias unit on hidden layer


    # positive phase

    # calculate hidden unit probabilties given the visibles (training vec)
    posprobs = np.tanh(np.dot(W.T, state[:nvisible]))
    # sample for the actual state
    state[nvisible:] = posprobs > np.random.rand(nhidden)
    # positive contribution
    positive = np.outer(state[:nvisible], state[nvisible:])

    # negative phase
    for k in xrange(cdk):
        # resample visible units
        visreconprobs = np.tanh(np.dot(W, state[nvisible:]))
        state[:nvisible] = visreconprobs > np.random.rand(nvisible)
        # resample hidden units
        negprobs = np.tanh(np.dot(W.T, visreconprobs))
        state[nvisible:] = negprobs > np.random.rand(1, nhidden)

    # negative contribution
    negative = np.outer(state[:nvisible], state[nvisible:])

    return positive - negative


@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
def train_restricted(np.float_t [:, :] data, 
                     np.ndarray[np.float_t, ndim=2] W,
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
        for idat in xrange(data.shape[1]):
            # initialize state to the training vector
            state[:nvisible] = data[:,idat]
            # print contr_div(state, W, nvisible, nhidden, cdk)
            # print contr_div(state, W, nvisible, nhidden, cdk)*eta
            # update weights with contrastive divergence
            W += contr_div(state, W, nvisible, nhidden, cdk)*eta
    return W

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
def sample_restricted(np.float_t [:] start, 
                      np.ndarray[np.float_t, ndim=2] W,
                      int ksteps):
    """
    This method implements a @k-step Gibbs sampler for
    the hidden units of a restricted Boltzmann machine
    given some starting vector @start. @W is the coupling 
    matrix.
    """
    cdef int k = 0
    cdef int nvisible = W.shape[0]
    cdef int nhidden = W.shape[1]
    cdef np.ndarray[np.float_t, ndim=1] state = np.zeros(nvisible+nhidden)

    for k in xrange(ksteps):
        # calculate hidden unit probabilties given the visibles (training vec)
        posprobs = np.tanh(np.dot(W.T, state[:nvisible]))
        # sample for the actual state
        state[:nvisible] = posprobs > np.random.rand(nhidden)
        # resample visible units
        visreconprobs = np.tanh(np.dot(W, state[nvisible:]))
        state[nvisible:] = visreconprobs > np.random.rand(nvisible)
        # resample hidden units
        negprobs = np.tanh(np.dot(W.T, visreconprobs))
        state[nvisible:] = negprobs > np.random.rand(1, nhidden)

    return state
