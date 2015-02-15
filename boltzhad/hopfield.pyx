'''

File: hopfield.pyx
Author: Hadayat Seddiqi
Date: 02.15.15
Description: Implement routines for training Hopfield networks.

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
cpdef train(np.float_t [:, :] data, 
            np.float_t [:, :] W, 
            int nsteps, 
            float eta, 
            float T, 
            int epochs, 
            rng):
    """
    Train a Hopfield network with visible units fully-connected
    on @data. The number of Monte Carlo steps in the Gibbs
    sampler is @nsteps, @T being the temperature, @eta the
    learning rate, and @epochs the number of times we loop
    through the data to train on (in random order each time).

    Data should be in a 2D matrix where columns represent training
    vectors and rows represent component variables.
    """
    cdef int ep = 0
    cdef int idat = 0
    cdef int istep = 0
    cdef int idxn = 0
    cdef float ediff = 0.0
    cdef int nb = 0
    cdef int neurons = W.shape[0]
    cdef np.float_t [:] state = np.empty(neurons)
    # for shuffling the unit indices in update loop
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(neurons))

    # training epochs
    for ep in xrange(epochs):
        # train W using MCMC for each training vector
        for idat in xrange(data.shape[1]):
            # initialize state to the training vector
            state = data[:,idat].copy()
            # with nogil:
            if 1:
                # chain steps
                for istep in xrange(nsteps):
                    # do one sweep over the network
                    for idxn in sidx_shuff:
                        # calculate energy difference
                        ediff = 0.0
                        for nb in xrange(neurons):
                            if nb != idxn:
                                ediff += 2.0*state[idxn]*(W[idxn,nb]*state[nb])
                        # decide to flip or not according to Gibbs update
                        if min(1,1./(1+cexp(ediff/T))) > crand()/float(RAND_MAX):
                            state[idxn] *= -1
                    # reshuffle spin-update indices
                    sidx_shuff = rng.permutation(sidx_shuff)
                # update the weights (difference between data and model sample)
                W = W + (np.outer(data[:,idat], data[:,idat]) - 
                         np.outer(state,state))*eta/2.0
                # make sure we have no self-connections
                # np.fill_diagonal(W, 0.0)
                W -= np.diag(W)
    return W
