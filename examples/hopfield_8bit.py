'''

File: 8bit.py
Author: Hadayat Seddiqi
Date: 02.15.15
Description: Test out some simple 8-neuron problems.

'''

import numpy as np
import boltzhad.hopfield as hopfield


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
datasp = np.asarray(
    rng.permutation([ bitstr2spins(v) for v in data ]),
    dtype=np.float).T
# num of neurons
neurons = datasp[:,0].size
# # covariance matrix
# K = np.cov(datasp.T)
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
# train the weights
W = hopfield.train(datasp, W, nsteps, eta, temp, epochs, rng)
# Print out data
print("Training data (in order given to model):")
for dvec in datasp.T:
    bvec = spins2bits(dvec)
    print(reduce(lambda x,y: x+y, [ str(k) for k in bvec ]))
# Print out energies from trained model
print("Lowest states and their energies:")
results = [0]*(2**neurons)
# enumerate all possible states and their energies
for ib, b in enumerate([ bin(x)[2:].rjust(neurons, '0') 
                         for x in range(2**neurons) ]):
    svec = np.asarray(
        bitstr2spins(np.array([ int(k) for k in b ])),
        dtype=np.float)
    results[ib] = [b, hopfield.isingenergy(svec, W)]
# sort and print the most likely ones
for res in sorted(results, key=lambda x:x[1])[:2*neurons]:
    if res[0] in data:
        print(res+['training vector'])
    else:
        print(res)
