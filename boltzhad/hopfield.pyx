'''

File: hopfield.pyx
Author: Hadayat Seddiqi
Date: 02.15.15
Description: Implement routines for training Hopfield networks.

'''

import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sps
import itertools
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

@cython.boundscheck(True)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef np.ndarray[np.float_t, ndim=2] train_sparse(
    np.float_t [:, :] data, 
    np.ndarray[np.float_t, ndim=2] W,
    int nsteps, 
    float eta, 
    float T, 
    int epochs, 
    rng
):
    """
    Train a Hopfield network with visible units fully-connected
    on @data. The number of Monte Carlo steps in the Gibbs
    sampler is @nsteps, @T being the temperature, @eta the
    learning rate, and @epochs the number of times we loop
    through the data to train on (in random order each time).

    Data should be in a 2D matrix where columns represent training
    vectors and rows represent component variables.

    This version takes in a scipy.sparse matrix for the weights @W
    and enforces this sparsity at each weight update.
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
    # check that the data vectors are the right length
    if W.shape[0] != data.shape[0]:
        print("Warning: data and weight matrix shapes don't match.")
    # create a mask to enforce sparsity quickly
    Wmask = W.copy()
    Wmask[Wmask != 0.0] = 1.0
    # training epochs
    for ep in xrange(epochs):
        # train W using MCMC for each training vector
        for idat in xrange(data.shape[1]):
            # initialize state to the training vector
            state = data[:,idat].copy()
            # chain steps
            for istep in xrange(nsteps):
                # do one sweep over the network
                for idxn in sidx_shuff:
                    # calculate energy difference
                    ediff = 0.0
                    for nb in xrange(neurons):
                        # we only have upper triangle of @W
                        if idxn < nb:
                            ediff += 2.0*state[idxn]*(W[idxn,nb]*state[nb])
                        elif idxn > nb:
                            ediff += 2.0*state[idxn]*(W[nb,idxn]*state[nb])
                    # decide to flip or not according to Gibbs update
                    if min(1,1./(1+cexp(ediff/T))) > crand()/float(RAND_MAX):
                        state[idxn] *= -1
                # reshuffle spin-update indices
                sidx_shuff = rng.permutation(sidx_shuff)
            # update the weights (difference between data and model sample)
            # we need to make sure we preserve the network topology
            W = W + np.multiply((np.outer(data[:,idat], data[:,idat]) - 
                                 np.outer(state,state))*eta/2.0,
                                Wmask)
    return W


def rand_2d_lattice(int nrows, 
                    int ncols, 
                    rng, 
                    int periodic=0):
    """
    Generate a 2D square Ising model on a torus (with periodic boundaries).
    Couplings are between [-1e-8,1e-8] randomly chosen from a uniform distribution.
    
    Note: in the code there is mixed usage of "rows" and "columns". @nrows
    talks about the number of rows in the lattice, but the jrow variable
    references the row index of the J matrix (i.e. a particular spin).

    Returns: Ising matrix in sparse DOK format
    """
    cdef int nspins = nrows*ncols
    cdef int jrow = 0
    cdef int jcol = 0
    # Generate periodic lattice adjacency matrix
    J = sps.dok_matrix((nspins,nspins), dtype=np.float64)
    for jrow in xrange(nspins):
        # periodic vertical (consider first "row" in square lattice only)
        if (jrow < ncols) and periodic:
            J[jrow, jrow + ncols*(nrows-1)] = rng.uniform(low=-1e-8, high=1e-8)
        # periodic horizontal
        if (jrow % ncols == 0.0) and periodic:
            J[jrow, jrow+ncols-1] = rng.uniform(low=-1e-8, high=1e-8)
        # loop through columns
        for jcol in xrange(jrow, nspins):
            # horizontal neighbors (we can build it all using right neighbors)
            if ((jcol == jrow + 1) and 
                (jrow % ncols != ncols - 1)):  # right neighbor
                J[jrow, jcol] = rng.uniform(low=-1e-8, high=1e-8)
            # vertical neighbors (we can build it all using bottom neighbors)
            if (jcol == jrow + ncols):
                J[jrow, jcol] = rng.uniform(low=-1e-8, high=1e-8)
    return J

def rand_kblock_2d_lattice(int nrows, 
                           int ncols, 
                           rng, 
                           int k=1):
    """
    Generate an Ising model that extends the 2D lattice case to where each
    spin has coupling to all other spins in a "block" of radius @k.

    @k = 1:
             o------o------o
             |      |      |
             |      |      |
             o-----|Z|-----o
             |      |      |
             |      |      |
             o------o------o

    @k = 2:

      o------o------o------o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o------o------o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o-----|Z|-----o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o------o------o------o
      |      |      |      |      |
      |      |      |      |      |
      o------o------o------o------o


    where each 'o' is directly coupled to 'Z', the central spin we're 
    considering. This forms a kind of receptive field around each neuron.
    Couplings are between [-1e-8,1e-8] randomly chosen from a uniform distribution.
    
    Returns: Ising matrix in sparse DOK format
    """
    cdef int nspins = nrows*ncols
    cdef int ispin = 0
    cdef int jspin = 0
    cdef int ki = 0
    cdef int tlc = 0
    cdef int indicator = 0

    # Generate periodic lattice adjacency matrix
    J = sps.dok_matrix((nspins,nspins), dtype=np.float64)
    # loop through spins
    for ispin in xrange(nspins):
        # loop over the radii (no self-connections, so no zero)
        for ki in xrange(1,k):
            # top left corner spin
            tlc = ispin - (ki+1)*ncols - 1 + ki
            # indicate whether any of these are valid neighbor spins
            indicator = 0
            # loop over all spins at ki-th radius
            # upper side
            for idx in (tlc+r for r in xrange(2*ki+1)):
                # check upper boundary (if violated, we're done)
                if (idx < 0):
                    break
                # check left and right boundary
                elif (idx % ncols == ncols - 1) or (idx % ncols == 1):
                    continue
                # ensure we populate upper triangle of J
                elif ispin < idx:
                    indicator = 1
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
            # lower side
            for idx in (tlc+r+2*ki*ncols for r in xrange(2*ki+1)):
                # check lower boundary (if violated, we're done)
                if (idx >= nspins):
                    break
                # check left and right boundary
                elif (idx % ncols == ncols - 1) or (idx % ncols == 1):
                    continue
                # ensure we populate upper triangle of J
                elif ispin < idx:
                    indicator = 1
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
            # left side
            for idx in (tlc+r*ncols for r in xrange(1,2*ki+1)):
                # check left boundary (if violated, we're done)
                if (idx % ncols == ncols - 1):
                    break
                # check upper and lower boundary
                elif (idx < 0) or (idx >= nspins):
                    continue
                # ensure we populate upper triangle of J
                elif ispin < idx:
                    indicator = 1
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
            # right side
            for idx in (tlc+2*ki+r*ncols for r in xrange(1,2*ki+1)):
                # check right boundary (if violated, we're done)
                if (idx % ncols == 1):
                    break
                # check upper and lower boundary
                elif (idx < 0) or (idx >= nspins):
                    continue
                # ensure we populate upper triangle of J
                elif ispin < idx:
                    indicator = 1
                    J[ispin, idx] = rng.uniform(low=-1e-8, high=1e-8)
            # if we found no valid spins at this radius, then the user
            # has overspecified @k since it's larger than the grid
            if not indicator:
                # readjust @k since user clearly screwed up
                # k = ki
                break
    return J
