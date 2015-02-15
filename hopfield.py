'''
'''

import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.optimize as spo
import pyximport; pyximport.install()


def isingenergy(z, J):
    """ Return the energy of the Hopfield network. """
    return -0.5*np.inner(z, np.inner(J, z))

def bits2spins(vec):
    """ Convert a bitvector @vec to a spinvector. """
    return [ -1 if k == 1 else 1 for k in vec ]

def spins2bits(vec):
    """ Convert a spinvector @vec to a bitvector. """
    return [ 0 if k == 1 else 1 for k in vec ]

def bitstr2spins(vec):
    """ Take a bitstring and return a spinvector. """
    a = [ int(k) for k in vec ]
    return bits2spins(a)

# Random number generator
seed = None
rng = np.random.RandomState(seed)
# training data
# 8-neuron network
data = ['10000001', 
        '10101010',
        # '00001111',
        '11100111']
# 20-neuron network
# data = ['10000000000000000001', 
#         '10101010101010101010',
#         '11100111001110011100']
datasp = rng.permutation([ bitstr2spins(v) for v in data ])
# num of neurons
neurons = datasp[0].size
# covariance matrix
K = np.cov(datasp.T)
# weight matrix (tiny random numbers)
W = rng.rand(neurons,neurons)*1e-8
# number of MCMC steps
nsteps = 10
# learning rate
eta = 0.01
# temperature
temp = 1.0
# training epochs
epochs = 10
# state vector
states = []
# training epochs
for ep in xrange(epochs):
    # train W using MCMC for each training vector
    for tvec in datasp:
        # initialize state to the training vector
        state = tvec.copy()
        # chain steps
        for istep in xrange(nsteps):
            # do one sweep over the network
            for idxn in rng.permutation(range(neurons)):
                # calculate energy difference
                ediff = 0.0
                for nb in xrange(neurons):
                    if nb != idxn:
                        ediff += 2.0*state[idxn]*(W[idxn,nb]*state[nb])
                # update rule
                alpha = min(1, 1./(1+np.exp(ediff/temp)))  # Gibbs
                # decide to flip or not, keep as new chain sample either way
                if alpha > rng.uniform(0,1):
                    state[idxn] *= -1
        # update the weights (difference between data and model sample)
        W += (np.outer(tvec,tvec) - np.outer(state,state))*eta
        # make sure we have no self-connections
        np.fill_diagonal(W, 0.0)

# Print out data
print("Training data (in order given to model):")
for dvec in datasp:
    bvec = spins2bits(dvec)
    print(reduce(lambda x,y: x+y, [ str(k) for k in bvec ]))
# Print out energies from trained model
print("Lowest states and their energies:")
results = [0]*(2**neurons)
# enumerate all possible states and their energies
for ib, b in enumerate([ bin(x)[2:].rjust(neurons, '0') 
                         for x in range(2**neurons) ]):
    svec = bitstr2spins(np.array([ int(k) for k in b ]))
    results[ib] = [b, isingenergy(svec, W)]
# sort and print the most likely ones
for res in sorted(results, key=lambda x:x[1])[:2*neurons]:
    if res[0] in data:
        print(res+['training vector'])
    else:
        print(res)
