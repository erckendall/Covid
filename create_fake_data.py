# Note, the transitions are random, so no penalty based on distance
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'all':lambda x: str(int(x))})

rt_size = 8
size = rt_size ** 2

# n0 = 1000 * np.random.rand(size)
n0 = [np.random.uniform(1000, 1500) for i in xrange(size)]
n0 = np.round(n0,0)


m1 = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        m1[i,j] = np.random.uniform(0,20)
m1 = np.round(m1, 0)
np.fill_diagonal(m1, 0)
m1_r = m1

# mij is people going from i to j
# mji is people going from j to i
# so n1[i] = n0[i] - sum_j of Mij + sum_j of Mji
# but we define nti = sum_j mijt

n1 = np.zeros(size)
for i in range(size):
    minus = 0
    plus = 0
    for j in range(size):
        minus += m1[i][j]
        plus += m1[j][i]
    n1[i] = n0[i] - minus + plus
    m1_r[i][i] = n0[i] - minus


m2 = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        m2[i,j] = np.random.uniform(0,20)
m2 = np.round(m2, 0)
np.fill_diagonal(m2, 0)
m2_r = m2

n2 = np.zeros(size)
for i in range(size):
    minus = 0
    plus = 0
    for j in range(size):
        minus += m2[i][j]
        plus += m2[j][i]
    n2[i] = n1[i] - minus + plus
    m2_r[i][i] = n1[i] - minus

m3 = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        m3[i,j] = np.random.uniform(0,20)
m3 = np.round(m3, 0)
np.fill_diagonal(m3, 0)
m3_r = m3

n3 = np.zeros(size)
for i in range(size):
    minus = 0
    plus = 0
    for j in range(size):
        minus += m3[i][j]
        plus += m3[j][i]
    n3[i] = n2[i] - minus + plus
    m3_r[i][i] = n2[i] - minus

if np.sum(n1) != np.sum(n2) or np.sum(n1) != np.sum(n3):
    print "Error: total not conserved"

if np.min(m1_r) < 0 or np.min(m2_r) < 0 or np.min(m3_r) < 0:
    print "Error: negative values in M matrices"

N = [n0, n1, n2, n3]
np.save('N.npy', N)
M = [m1_r, m2_r, m3_r]
np.save('M.npy', M)


loc_list = []
for i in range(rt_size):
    for j in range(rt_size):
        loc_list.append([i, j])

dist_mat = np.zeros((size,size))
for k in range(size):
    for l in range(size):
        dist_mat[k][l] = np.sqrt((loc_list[k][0] - loc_list[l][0])**2 + (loc_list[k][1] - loc_list[l][1])**2)

np.save('dist_fake.npy', dist_mat)



