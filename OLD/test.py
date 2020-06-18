import numpy as np

vals = np.load('M_00001.npy')

np.set_printoptions(formatter={'all':lambda x: str(int(x))})
print np.round(vals[0], 0)