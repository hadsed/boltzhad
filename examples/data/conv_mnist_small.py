'''
'''

import numpy as np
import scipy.io as sio

# data
d = sio.loadmat('mnist_all.mat')
# relevant keys
keys = ['test1', 'test0', 'test3', 'test2', 'test5', 'test4', 
        'test7', 'test6', 'test9', 'test8', 'train4', 'train5', 
        'train6', 'train7', 'train0', 'train1', 'train2', 
        'train3', 'train8', 'train9']
# new data dict
newd = dict((k, np.array([])) for k in keys)
# loop through
for k in keys:
    hold = []
    for p in d[k]:
        t = p.reshape(28,28)
        hold.append(t[::2,::2].reshape(14*14))
    newd[k] = np.array(hold)

sio.savemat('mnist_small.mat', newd)
