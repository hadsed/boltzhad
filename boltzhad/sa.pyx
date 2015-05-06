# encoding: utf-8
# cython: profile=False
# filename: sa.pyx
'''

File: sa.pyx
Author: Hadayat Seddiqi
Date: 10.13.14
Description: Do thermal annealing on a (sparse) Ising system.

'''

import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange
from libc.math cimport exp as cexp
from libc.stdlib cimport rand as crand
from libc.stdlib cimport RAND_MAX as RAND_MAX
# from libc.stdio cimport printf as cprintf


def bits2spins(vec):
    """ Convert a bitvector @vec to a spinvector. """
    return np.asarray([ -1 if k == 1 else 1 for k in vec ],
                      dtype=np.float)

def spins2bits(vec):
    """ Convert a spinvector @vec to a bitvector. """
    return np.asarray([ 0 if k == 1 else 1 for k in vec ], 
                      dtype=np.float)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef IsingEnergy(np.float_t[:] spins, 
                  np.float_t[:, :] J):
    """
    Calculate energy for Ising graph @J in configuration @spins.
    Generally not needed for the annealing process but useful to
    have around at the end of simulations.
    """
    cdef np.float_t[:, :] d = np.zeros((spins.size,spins.size))
    # J = np.asarray(J.todense())
    J = np.asarray(J)
    d = np.diag(np.diag(J))
    J = J - np.diag(J)
    return -np.dot(spins, np.dot(J, spins)) - np.sum(np.dot(d,spins))

cpdef GenerateNeighbors(int nspins, 
                        J,  # scipy.sparse matrix
                        int maxnb, 
                        str savepath=None):
    """
    Precompute a list that include neighboring indices to each spin
    and the corresponding coupling value. Specifically, build:

    neighbors = [
           [ [ ni_0, J[0, ni_0] ], 
             [ ni_1, J[0, ni_1] ], 
               ... ],

           [ [ ni_0, J[1, ni_0] ], 
             [ ni_1, J[1, ni_1] ], 
               ... ],

            ...

           [ [ ni_0, J[nspins-1, ni_0]], 
             [ ni_1, J[nspins-1, ni_1]],                   
               ... ]
     ]

    For graphs that are not completely "regular", there will be
    some rows in the neighbor matrix for each spin that will show
    [0,0]. This is required to keep the neighbors data structure
    an N-dimensional array, but in the energy calculations will have
    no contribution as the coupling strength is essentially zero.
    On the other hand, this is why @maxnb must be set to as high a
    number as necessary, but no more (otherwise it will incur some
    computational cost).

    Inputs:  @npsins   number of spins in the 2D lattice
             @J        Ising coupling matrix
             @maxnb    the maximum number of neighbors for any spin
                       (if self-connections representing local field
                       terms are present along the diagonal of @J, 
                       this counts as a "neighbor" as well)

    Returns: the above specified "neighbors" list as a numpy array.
    """
    # predefining vars
    cdef int ispin = 0
    cdef int ipair = 0
    # the neighbors data structure
    cdef np.float_t[:, :, :]  nbs = np.zeros((nspins, maxnb, 2))
    # dictionary of keys type makes this easy
    J = J.todok()
    # Iterate over all spins
    for ispin in xrange(nspins):
        ipair = 0
        # Find the pairs including this spin
        for pair in J.iterkeys():
            if pair[0] == ispin:
                nbs[ispin, ipair, 0] = pair[1]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
            elif pair[1] == ispin:
                nbs[ispin, ipair, 0] = pair[0]
                nbs[ispin, ipair, 1] = J[pair]
                ipair += 1
    J = J.tocsr()  # DOK is really slow for multiplication
    if savepath is not None:
        np.save(savepath, nbs)
    return nbs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef Anneal(np.float_t[:] sched, 
             int mcsteps, 
             np.float_t[:] svec, 
             np.float_t[:, :, :] nbs, 
             rng):
    """
    Execute thermal annealing according to @annealingSchedule, an
    array of temperatures, which takes @mcSteps number of Monte Carlo
    steps per timestep.

    Starting configuration is given by @spinVector, which we update 
    and calculate energies using the Ising graph @isingJ. @rng is the 
    random number generator.

    Returns: None (spins are flipped in-place)
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef float ediff = 0.0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))

    # Loop over temperatures
    for itemp in xrange(sched.size):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over spins
            for sidx in sidx_shuff:
                # loop through the given spin's neighbors
                for si in xrange(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[sidx,si,0])
                    # get the coupling value to that neighbor
                    jval = nbs[sidx,si,1]
                    # self-connections are not quadratic
                    if spinidx == sidx:
                        ediff += -2.0*svec[sidx]*jval
                    # calculate the energy diff of flipping this spin
                    else:
                        ediff += -2.0*svec[sidx]*(jval*svec[spinidx])
                # Metropolis accept or reject
                if ediff > 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
@cython.cdivision(True)
cpdef Anneal_dense(np.float_t[:] sched, 
                   int mcsteps, 
                   np.float_t[:] svec, 
                   np.float_t[:, :] J, 
                   rng):
    """
    Execute thermal annealing according to @annealingSchedule, an
    array of temperatures, which takes @mcSteps number of Monte Carlo
    steps per timestep.

    Starting configuration is given by @spinVector, which we update 
    and calculate energies using the Ising graph @isingJ. @rng is the 
    random number generator.

    Returns: None (spins are flipped in-place)
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int step = 0
    cdef int sidx = 0
    cdef int si = 0
    cdef float ediff = 0.0
    cdef np.ndarray[np.int_t, ndim=1] sidx_shuff = \
        rng.permutation(range(nspins))

    # Loop over temperatures
    for itemp in xrange(sched.size):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over spins
            for sidx in sidx_shuff:
                # loop through the given spin's neighbors
                for si in xrange(nspins):
                    # print("yahoo3", sidx, si, ediff)
                    # print(J[sidx,si], svec[sidx], svec[si])
                    # self-connections are not quadratic
                    if si == sidx:
                        ediff += -2.0*svec[sidx]*J[sidx,si]
                    # calculate the energy diff of flipping this spin
                    else:
                        # incase we only have upper triangle
                        if sidx < si:
                            ediff += -2.0*svec[sidx]*(J[sidx,si]*svec[si])
                        elif sidx > si:
                            ediff += -2.0*svec[sidx]*(J[si,sidx]*svec[si])
                # print("yahoo4")
                # Metropolis accept or reject
                if ediff > 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediff/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
                # Reset energy diff value
                ediff = 0.0
            sidx_shuff = rng.permutation(sidx_shuff)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef Anneal_parallel(np.float_t[:] sched, 
                      int mcsteps, 
                      np.float_t[:] svec, 
                      np.float_t[:, :, :] nbs, 
                      int nthreads):
    """
    Execute thermal annealing according to @annealingSchedule, an
    array of temperatures, which takes @mcSteps number of Monte Carlo
    steps per timestep.

    Starting configuration is given by @spinVector, which we update 
    and calculate energies using the Ising graph @isingJ.

    This version attempts to do thread parallelization with Cython's
    built-in OpenMP directive "prange". The extra argument @nthreads
    specifies how many workers to split the spin updates amongst.

    Note that while the sequential version randomizes the order of
    spin updates, this version does not.

    Returns: None (spins are flipped in-place)
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int maxnb = nbs[0].shape[0]
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int sidx = 0
    cdef int si = 0
    cdef int spinidx = 0
    cdef float jval = 0.0
    cdef np.ndarray[np.float_t, ndim=1] ediffs = np.zeros(nspins)

    # Loop over temperatures
    for itemp in xrange(sched.size):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over spins
            # print nthreads, openmp.omp_get_num_threads()
            for sidx in prange(nspins, nogil=True, 
                               schedule='guided', 
                               num_threads=nthreads):
                # loop through the neighbors
                for si in xrange(maxnb):
                    # get the neighbor spin index
                    spinidx = int(nbs[sidx, si, 0])
                    # get the coupling value to that neighbor
                    jval = nbs[sidx, si, 1]
                    # self-connections are not quadratic
                    if spinidx == sidx:
                        ediffs[sidx] += -2.0*svec[sidx]*jval
                    else:
                        ediffs[sidx] += -2.0*svec[sidx]*(jval*svec[spinidx])
                # Accept or reject
                if ediffs[sidx] > 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediffs[sidx]/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
            # reset
            ediffs.fill(0.0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef Anneal_dense_parallel(np.float_t[:] sched, 
                            int mcsteps, 
                            np.float_t[:] svec, 
                            np.float_t[:, :] J, 
                            int nthreads):
    """
    Execute thermal annealing according to @annealingSchedule, an
    array of temperatures, which takes @mcSteps number of Monte Carlo
    steps per timestep.

    Starting configuration is given by @spinVector, which we update 
    and calculate energies using the Ising graph @isingJ.

    This version attempts to do thread parallelization with Cython's
    built-in OpenMP directive "prange". The extra argument @nthreads
    specifies how many workers to split the spin updates amongst.

    Note that while the sequential version randomizes the order of
    spin updates, this version does not.

    Returns: None (spins are flipped in-place)
    """
    # Define some variables
    cdef int nspins = svec.size
    cdef int itemp = 0
    cdef float temp = 0.0
    cdef int sidx = 0
    cdef int si = 0
    cdef np.ndarray[np.float_t, ndim=1] ediffs = np.zeros(nspins)

    # Loop over temperatures
    for itemp in xrange(sched.size):
        # Get temperature
        temp = sched[itemp]
        # Do some number of Monte Carlo steps
        for step in xrange(mcsteps):
            # Loop over spins
            # print nthreads, openmp.omp_get_num_threads()
            for sidx in prange(nspins, nogil=True, 
                               schedule='guided', 
                               num_threads=nthreads):
                # loop through the neighbors
                for si in xrange(nspins):
                    # self-connections are not quadratic
                    if si == sidx:
                        ediffs[sidx] += -2.0*svec[sidx]*J[sidx,si]
                    else:
                        # incase we only have upper triangle
                        if sidx < si:
                            ediffs[sidx] += -2.0*svec[sidx]*(J[sidx,si]*svec[si])
                        elif sidx > si:
                            ediffs[sidx] += -2.0*svec[sidx]*(J[si,sidx]*svec[si])
                # Accept or reject
                if ediffs[sidx] > 0.0:  # avoid overflow
                    svec[sidx] *= -1
                elif cexp(ediffs[sidx]/temp) > crand()/float(RAND_MAX):
                    svec[sidx] *= -1
            # reset
            ediffs.fill(0.0)
