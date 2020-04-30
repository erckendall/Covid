import numpy as np

mat = np.random.rand(4,4) * 10
mat = np.around(mat, 0)

simp = []
for i in range(4):
    simp.append(i)



simp_2 = np.ones((1,4))
for i in range(4):
    simp_2[0,i] = i
print (simp_2*mat)[0]


matlist = []
for i in range(2):
    matlist.append(mat[i])
print np.transpose(np.transpose(matlist[0])*simp)


for i in range(4):
    print dist[i]*s[]

tolog = []
for i in range(lim):


snd = -np.sum(np.sum(Y_trunc, axis=0) * np.log(np.sum(s_1 * np.exp(-beta * dist) * ar, axis=1)))


# sk times beta ik,

# tot = 0
# for i in range(n_tsteps):
#     val = np.transpose(np.transpose(X_trunc[i]) * np.log(s))
#     tot += np.sum(val)
