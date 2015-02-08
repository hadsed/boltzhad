'''
'''

import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.optimize as spo


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
data = ['10000001', 
        '10101010']
data = np.array([ bitstr2spins(v) for v in data ])
# num of neurons
neurons = data[0].size
# covariance matrix
K = np.cov(data.T)
# weight matrix (tiny random numbers)
W = rng.rand(neurons,neurons)*1e-3
# number of MCMC steps
nsteps = 100
# learning rate
eta = 0.1
# temperature
temp = 1e-1
# state vector
states = []
# train W using MCMC for each training vector
for tvec in data:
    # initialize state to the training vector
    state = tvec
    # chain steps
    for istep in xrange(nsteps):
        # do one sweep over the network
        for idxn in rng.permutation(range(neurons)):
            # calculate energy difference
            state_c = state.copy()
            state_c[idxn] *= -1
            ediff = isingenergy(state_c,W) - isingenergy(state,W)
            # update rule
            # alpha = min(1, np.exp(ediff/temp))  # Metropolis
            alpha = min(1, 1./(1+np.exp(ediff/temp)))  # Gibbs
            # decide to flip or not, keep as new chain sample either way
            if alpha > rng.uniform(0,1):
                states.append(state_c)
            else:
                states.append(state)
    # make sure we only take one sample per sweep (on avg)
    Zexp = np.sum([ np.outer(s,s) for s in states[::neurons] ], 
                  axis=0)/len(states)
    # update the weights
    W += (Zexp + K)*eta

# Print out data
print("Training data:")
for dvec in data:
    bvec = spins2bits(dvec)
    print(reduce(lambda x,y: x+y, [ str(k) for k in bvec ]))
# Print out energies from trained model
print("All possible states and their energies:")
results = []
for b in [ bin(x)[2:].rjust(neurons, '0') for x in range(2**neurons) ]:
    bvec = np.array([ int(k) for k in b ])
    svec = bitstr2spins(b)
    bstr = reduce(lambda x,y: x+y, [ str(k) for k in bvec ])
    results.append([bstr, isingenergy(svec, W)])
for res in sorted(results, key=lambda x:x[1])[:8]:
    print res
for res in sorted(results, key=lambda x:x[1])[-5:]:
    print res
        
