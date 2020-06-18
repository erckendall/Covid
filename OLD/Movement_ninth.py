import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
import timeit
warnings.filterwarnings("ignore")
import numba


### Loading the Southland names, data, and distance matrix
names = np.load('southland_names.npy').tolist()
vals = np.load('southland_vals.npy').tolist()
dist = np.load('southland_distance_matrix.npy')

### Delete 2 regions for which data is incomplete (both average <10 people)
to_del = []
ind = -1
for i in vals:
    ind += 1
    if len(i) != 2856:
        to_del.append(ind)
names.pop(to_del[0])
names.pop(to_del[1] - 1)
vals.pop(to_del[0])
vals.pop(to_del[1] - 1)
dist = np.delete(dist, to_del[0], 0)
dist = np.delete(dist, (to_del[1] - 1), 0)
dist = np.delete(dist, to_del[0], 1)
dist = np.delete(dist, (to_del[1] - 1), 1)

### For testing, work with small subset of regions
lim = 10
names_test = []
vals_test = []
dist_test = np.zeros((lim, lim))
for i in range(lim):
    names_test.append(names[i])
    vals_test.append(vals[i])
    for j in range(lim):
        dist_test[i, j] = dist[i, j]
names = names_test
vals = vals_test
dist = dist_test



### Want to include filter s.t. regions more than 80km cannot be accessed in one timestep
### use np.where on dist matrix - set to zeros if greater, ones if smaller

K_cut = np.where(dist > 80, 0, 1)
print K_cut


### Looking at just Feb 2020 data:
count_f = len(filter(lambda n: (int(n[1]) == 2 and int(n[0]) == 2020), vals[0]))
count_j = len(filter(lambda n: (int(n[1]) == 1 and int(n[0]) == 2020), vals[0]))
n_tsteps = count_f - 1 # -1 because number of timesteps one less than number of times ## Check if this is what paper means


### Ordering the data
ordered_vals_f_2020 = []
for val in vals:
    ordered = np.zeros(n_tsteps + 1)
    ftd = filter(lambda n: (int(n[0]) == 2020 and int(n[1]) == 2), val)
    for i in ftd:
        ordered[(int(i[2]) - 1) * 24 + int(i[3])] = int(i[4])
    ordered_vals_f_2020.append(ordered.tolist())


### Optional redefine number of timesteps e.g. only look at 12 hour section
n_tsteps = 4


##################### Algorithm to estimate movements from static data (approximate inference)

###Initialise parameters. Note initial assumptions for M, beta, lmbda - evaluate sensitivity to these
start = timeit.default_timer()
Cut, N, M, X, Y, Z, pi, s = ([] for i in range(8))
theta, mu = (np.zeros((len(names), len(names))) for i in range(2))
beta = 0. #was 0.
lmbda = 1. #was 10.

### For each timestep we create an N_t list and an  M_t matrix
for t in range(n_tsteps + 1):  # originally had  +1 here, but think by timesteps the authours actually mean times
    N_t = []
    for val in ordered_vals_f_2020:
        N_t.append(val[t])
        M_t = np.ones((len(names), len(names)))
    N.append(N_t)
    M.append(M_t * 10)
    Cut.append(K_cut)


### Initialise s for each region (not a function of time)
for i in range(len(names)):
    s.append(0.5)


### For each timestep, initialise list of X_t_i_j , Y_t_i and Z_t_i for each i in names
for t in range(n_tsteps + 1):
    X_t, Y_t, Z_t = ([] for i in range(3))
    for i in range(len(names)):
        X_t_i_j = []
        Y_t_i = 0
        Z_t.append((M[t])[i][i])
        for j in range(len(names)):             ### Implement K_cut here?
            X_t_i_j.append((M[t])[j][i])
            if i != j:
                Y_t_i += (M[t])[i][j]
        X_t.append(X_t_i_j)
        Y_t.append(Y_t_i)
    X.append(X_t)
    Y.append(Y_t)
    Z.append(Z_t)


### define f_s_b update (eq 11 to update s and beta) Need to maximise
@numba.jit(parallel=True)
def f_s_b(sbeta):
    beta = sbeta[-1]
    s = sbeta[0:lim]
    ar = np.ones((lim, lim))
    np.fill_diagonal(ar, 0)
    fst = np.sum(X_trunc * np.log(np.array(s))[np.newaxis, :, np.newaxis])
    snd = -np.sum(np.sum(Y_trunc, axis=0) * np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-beta * dist) * ar, axis=1)))
    thd = -beta * np.sum(dist[np.newaxis, :] * X_trunc)
    return -(fst+snd+thd)#, fst, snd, thd ### negative here as we want to maximise the positive value (minimise negative)




### Define approximate log likelihood function
@numba.jit(parallel=True)
def loglik(comb):
    for t in range(n_tsteps-2):
        for i in range(len(names)):
            Y_trunc[t][i] = comb[Y_plus + t * dy[1] + i]
            Z_trunc[t][i] = comb[Z_plus + t * dz[1] + i]
            for j in range(len(names)):
                X_trunc[t][i][j] = comb[t * dx[2] * dx[1] + i * dx[2] + j]
    np.array(Y_trunc)
    np.array(Z_trunc)
    np.array(X_trunc)
    pi_in = np.ones((lim))
    for i in range(len(names)):
        pi_in[i] = 1 - pi[i]
    fst = np.sum(Y_trunc * np.log(N_trunc * np.array(pi)[np.newaxis, :]) + Y_trunc - Y_trunc * np.log(Y_trunc))
    snd = np.sum(Z_trunc * np.log(N_trunc * pi_in[np.newaxis, :]) + Z_trunc - Z_trunc * np.log(Z_trunc))
    thd = np.sum(X_trunc * np.log(np.array(mu)[np.newaxis, :]) + X_trunc - X_trunc * np.log(X_trunc))
    first = abs(np.sum((N_trunc - Y_trunc - Z_trunc), axis=1))**2
    second = abs(np.sum((N_trunc_p1 - Z_trunc -np.sum(X_trunc, axis=2)), axis=1))**2
    tot = np.sum(first + second)
    return -(fst + snd + thd - (lmbda/2) * tot)




##################### Begin while here:
# ### Maximise loglik - lmbda/2*penalty. Implement constraints X,Y,Z >= 0
# ### Update pi with new  X Y Z  using eq. 10
# ### Maximise f_s_b - update s and beta
# ### Update theta and thereforemu with new s and pi values

current = 0.
conv = False
N_trunc = (np.array(N))[0:(n_tsteps - 2)]
N_trunc_p1 = (np.array(N))[1:(n_tsteps - 1)]

while conv == False:
    existing = current

    Y_trunc = (np.array(Y))[0:(n_tsteps-2)]
    Z_trunc = (np.array(Z))[0:(n_tsteps - 2)]
    X_trunc = (np.array(X))[0:(n_tsteps - 2)]

    ### pi update (eqn 10)
    pi = (np.sum(Y_trunc, axis=0) / (np.sum(Y_trunc, axis=0) + np.sum(Z_trunc, axis=0))).tolist()

    ### Calculate theta matrix (not time dependent) (eqn 2)
    for i in range(len(names)):
        sum = 0
        for j in range(len(names)):
            if j != i:
                sum += s[j] * np.exp(-beta * dist[i, j])
        for j in range(len(names)):
            if j == i:
                theta[i, j] = 1 - pi[i]
            else:
                theta[i, j] = pi[i] * (s[j] * np.exp(-beta * dist[i, j])) / (sum)

    ### Calculate mu given theta (assumes no equal distances)
    for i in range(len(names)):
        for j in range(len(names)):
            sum = 0
            for t in range(n_tsteps-2):
                sum += N[t][j]
            mu[i, j] = sum * theta[j][i] ### Weird - what is t here

    ### flattening and concatenating X, Y, Z to feed into optimisation
    dx = np.array(X).shape
    dy = np.array(Y).shape
    dz = np.array(Z).shape

    Xarr = np.array(X).flatten()
    Y_plus = Xarr.shape[0]  # may need -1
    Yarr = np.array(Y).flatten()
    Z_plus = Yarr.shape[0] + Xarr.shape[0]  # may need -2
    Zarr = np.array(Z).flatten()

    comb = np.concatenate((Xarr, Yarr, Zarr,))

    ### L-BFGS-B method doesn't respect bounds so need to modify
    bnds = []
    for i in range(comb.shape[0]):
        bnds.append((0, None))
    opt = {'maxiter': 1000, 'ftol': 16}
    newvals = scipy.optimize.minimize(fun=loglik, x0=comb, method="SLSQP", bounds=bnds, options=opt, tol= 16)
    try:
        assert newvals.success
    except AssertionError as err:
        print("newvals error ", newvals.message)
        print(err)
    XYZ = newvals.x
    current = newvals.fun


    print 'Current = ', current
    for i in XYZ:
        if i < 0:
            print "XYZ bounds exceeded"

    ### Putting X, Y, Z back into original format
    for t in range(n_tsteps):
        for i in range(len(names)):
            Y[t][i] = XYZ[Y_plus + t * dy[1] + i]
            Z[t][i] = XYZ[Z_plus + t * dz[1] + i]
            for j in range(len(names)):
                X[t][i][j] = XYZ[t * dx[2] * dx[1] + i * dx[2] + j]

    ### Flattening and concatenating s and beta for optimisation
    sarr = np.array(s).flatten()
    beta = np.array([beta])
    sbeta = np.concatenate((sarr, beta))

    ### Maximising f_s_b
    ### Note: depending on whether fsb update before or after XYZ update there is different behaviour
    bnds = []
    for i in range(sbeta.shape[0]):
        bnds.append((0, None))  ### s and beta bounds

    newvals = scipy.optimize.fmin_l_bfgs_b(f_s_b, sbeta, approx_grad=True, bounds=bnds, epsilon=1e-8, maxiter=20000, factr=1e7, pgtol=1.6)
    try:
        assert newvals[2]['warnflag'] == 0
    except AssertionError as err:
        print("beta error ", newvals[2]['task'])
        print(err)
    newsbeta = newvals[0]

    ### Putting s and beta back into original form
    for i in range(newsbeta.shape[0] - 1):
        s[i] = newsbeta[i]
    beta = newsbeta[-1]
    print 's = ', s
    print 'beta = ', beta
    for sval in s:
        if sval < 0:
            print "f_s_b bounds exceeded"

    if abs((existing - current)/current)*100 < .1:
        print "Converged to within 0.1%"
        conv = True


##################### End while

##################### Final calculation of M
### Flattening for optimisation
Marr = np.array(M)
dM = Marr.shape
Marr = Marr.flatten()
Cutarr = np.array(Cut).flatten()
ind_lst = []
for i in range(Cutarr.shape[0]):
    if Cutarr[i] == 0:
        ind_lst.append(i)

### Final pi update (eqn 10) - can do final s and beta as well but likely converged
pi_f = []
for i in range(len(names)):
    num = 0
    denom = 0
    for t in range(n_tsteps - 2):
        num += Y[t][i]
        denom += Y[t][i] + Z[t][i]
    pi_f.append((num / denom))
pi_f = np.array(pi_f)

### Exact log likelihood (eqn 4) ### Need to figure out how K_cut can help here - currently just imposed in initial values, not strict rule
### Probably needs to go into the bounds in the optimisation
### Also needs to go into the XYZ bounds - check defs

@numba.jit(parallel=True)
def loglik_ex(Marr):
    M_trunc = np.zeros((n_tsteps - 2, lim, lim))
    M_trunc_t = np.zeros((n_tsteps - 2, lim, lim))
    for t in range(n_tsteps-2):
        for i in range(len(names)):
            for j in range(len(names)):
                M_trunc[t][i][j] = (Marr[t * dM[2] * dM[1] + i * dM[2] + j])
                M_trunc_t[t][j][i] = (Marr[t * dM[2] * dM[1] + i * dM[2] + j])
    np.array(M_trunc)
    np.array(M_trunc_t)

    pi_in_log = np.zeros((lim))
    for h in range(len(names)):
        pi_in_log[h] = np.log(1 - pi_f[h])
    o_diag = np.ones((lim,lim))
    diag = np.zeros((lim, lim))
    np.fill_diagonal(o_diag, 0)
    np.fill_diagonal(diag, 1)

    fst = np.sum(abs(np.sum(N_trunc - np.sum(M_trunc, axis=2), axis=1))**2)
    snd = np.sum(abs(np.sum(N_trunc_p1 - np.sum(M_trunc_t, axis=2), axis=1))**2)

    first = pi_in_log[np.newaxis, np.newaxis, :] * M_trunc * diag[np.newaxis, :, :]
    mult = np.log(pi_f[:, np.newaxis]) + np.log(np.array(s)[:, np.newaxis]) - beta * dist
    mult2 = - np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-beta * dist) * o_diag, axis=1))
    second = M_trunc * o_diag[np.newaxis, :, :] * mult[np.newaxis, :, :] + M_trunc * o_diag[np.newaxis, :, :] * mult2[np.newaxis, :, np.newaxis]
    third = (M_trunc - M_trunc * np.log(M_trunc))*o_diag[np.newaxis, :, :]
    return -(np.sum(first + second + third) - (lmbda/2.) * (fst + snd))

### Performing final optimisation
bnds = []
for i in range(Marr.shape[0]):
    bnds.append((0., None))
### Note SLSQP method doesn't respect bounds so I use constraints instead

cons = []
for a in ind_lst:
    def con(Marr, a):
        return Marr[a]
    cons.append({'type': 'eq', 'fun': con, 'args': [a]})


opt = {'maxiter': 1000, 'ftol': 2.5} ### Note that implementing this seems to severely change M output
Mvals = scipy.optimize.minimize(fun=loglik_ex, x0=Marr, method="SLSQP", bounds=bnds, constraints=cons, options=opt, tol=2.5)
Mfinal = Mvals.x
print 'Loglik: ', Mvals.fun
try:
    assert Mvals.success
except AssertionError as err:
    print("Mvals error ", Mvals.message)
    print(err)


## Returning M matrices to original form
for t in range(n_tsteps):
    for i in range(len(names)):
        for j in range(len(names)):
            M[t][i][j] = int(Mfinal[t * dM[2] * dM[1] + i * dM[2] + j])


stop = timeit.default_timer()
np.set_printoptions(formatter={'all':lambda x: str(int(x))})
print '---------------------------'
print 'Number of regions = ', lim
print 'Number of timesteps = ', n_tsteps
print 'beta = ', beta
print 's = ', s
print 'M(t=0) = \n', M[0]
print 'Run time = ', ((stop - start)/60), 'mins'

###################### END




#### Attempt at using multiprocessing
# ### trying to use multiprocessing to speed up the above optimization
# def minim(comb):
#     newvals = scipy.optimize.minimize(fun=loglik, x0=comb, method="SLSQP", bounds=bnds, options=opt)
#     return newvals
# if __name__ == '__main__':
#         p = Pool(2)
#         res = (p.map(minim, (comb,)))
# XYZ = res[0].x
# current = res[0].fun
