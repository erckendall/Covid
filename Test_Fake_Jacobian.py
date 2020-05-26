import numpy as np
import sys
import os
import scipy.optimize
import warnings
import timeit
warnings.filterwarnings("ignore")
import numba
from numba import prange
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(formatter={'all':lambda x: str(int(x))})


lim = 9
run = 1000
lmbda = 1.
steps = 3
nv_factr = 1e7
nv_pgtol = 1.
beta_factr = 1e7
beta_pgtol = 1.
Mv_factr = 1e7
Mv_pgtol = 1.


app_list = []
f_list = []
beta_list = []


N = np.load('Fake_9/N.npy')
M_true = np.load('Fake_9/M.npy')
dist = np.load('Fake_9/dist_fake.npy')

### Want to include filter s.t. regions more than 80km cannot be accessed in one timestep
K_cut = np.where(dist > 1000, 0, 1)

### Optional redefine number of timesteps e.g. only look at 12 hour section
n_tsteps = steps


###Initialise parameters. Note initial assumptions for M, beta, lmbda - evaluate sensitivity to these
start = timeit.default_timer()
Cut, M, X, Y, Z, pi, s = ([] for i in range(7))
theta, mu = (np.zeros((lim, lim)) for i in range(2))
beta = 0.5

for t in range(n_tsteps):
    Cut.append(K_cut)

for t in range(n_tsteps):  # originally had  +1 here, but think by timesteps the authours actually mean time points
    M_t = np.random.rand(lim, lim)
    for i in range(lim):
        M_t[i][i] = N[t][i]
    M.append(M_t)


### Initialise s for each region (not a function of time)
for i in range(lim):
    s.append(1.)

### For each timestep, initialise list of X_t_i_j , Y_t_i and Z_t_i for each i in names
for t in range(n_tsteps):
    X_t, Y_t, Z_t = ([] for i in range(3))
    for i in range(lim):
        X_t_i_j = []
        Y_t_i = 0
        Z_t.append((M[t])[i][i])
        for j in range(lim):
            X_t_i_j.append((M[t])[j][i])
            if i != j:
                Y_t_i += (M[t])[i][j]
        X_t.append(X_t_i_j)
        Y_t.append(Y_t_i)
    X.append(X_t)
    Y.append(Y_t)
    Z.append(Z_t)


### defne analytic jacobian for approx log likelihood function
@numba.jit(parallel=True, nopython=True, nogil=True, cache=True)
def jacob(comb, mu, pi, N_trunc):
    Y_dev = Y_trunc.copy()
    Z_dev = Z_trunc.copy()
    X_dev = X_trunc.copy()
    for t in range(n_tsteps):
        for i in range(lim):
            Y_dev[t][i] = np.log(N_trunc[t][i] * pi[i]) - np.log(comb[Y_plus + t * dy[1] + i])
            Z_dev[t][i] = np.log(N_trunc[t][i] * (1 - pi[i])) - np.log(comb[Z_plus + t * dz[1] + i])
            for j in range(lim):
                X_dev[t][i][j] = np.log(mu[i][j]) - np.log(comb[t * dx[2] * dx[1] + i * dx[2] + j])
    X_dev = X_dev.flatten()
    Z_dev = Z_dev.flatten()
    Y_dev = Y_dev.flatten()
    jac = np.concatenate((X_dev, Y_dev, Z_dev,))
    return jac


### define f_s_b update (eq 11 to update s and beta) Need to maximise
@numba.jit(parallel=True)
def f_s_b(sbeta, Y_trunc, X_trunc):
    beta = sbeta[-1]
    s = sbeta[0:lim]
    ar = np.ones((lim, lim))
    np.fill_diagonal(ar, 0)
    fst = np.sum(X_trunc * np.log(np.array(s))[np.newaxis, :, np.newaxis])
    snd = -np.sum(np.sum(Y_trunc, axis=0) * np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-beta * dist) * ar, axis=1)))
    thd = -beta * np.sum(dist[np.newaxis, :] * X_trunc)
    return -(fst)#+snd+thd) ### negative here as we want to maximise the positive value (minimise negative)


### Define approximate log likelihood function
@numba.jit(parallel=True, nopython=True, nogil=True, cache=True)
def loglik(comb, mu, pi, N_trunc):
    Y_trun = Y_trunc.copy()
    Z_trun = Z_trunc.copy()
    X_trun = X_trunc.copy()

    for t in range(n_tsteps):
        for i in range(lim):
            Y_trun[t][i] = comb[Y_plus + t * dy[1] + i]
            Z_trun[t][i] = comb[Z_plus + t * dz[1] + i]
            for j in range(lim):
                X_trun[t][i][j] = comb[t * dx[2] * dx[1] + i * dx[2] + j]

    fst = np.sum(Y_trun * log_N_pi + Y_trun - Y_trun * np.log(Y_trun))
    snd = np.sum(Z_trun * log_N_pi_in + Z_trun - Z_trun * np.log(Z_trun))
    thd = np.sum(X_trun * log_mu_ext_ones / ones + X_trun - X_trun * np.log(X_trun))
    first = np.abs(np.sum((N_trunc - Y_trun - Z_trun), axis=1))**2
    second = np.abs(np.sum((N_trunc_p1 - Z_trun -np.sum(X_trun, axis=2)), axis=1))**2
    tot = np.sum(first + second)
    return -(fst + snd + thd - (lmbda/2) * tot)


##################### Begin while here:


current = 0.
current_2 = 0.
conv = False
N_trunc = (np.array(N))[0:(n_tsteps)]
N_trunc_p1 = (np.array(N))[1:(n_tsteps + 1)]

while conv == False:
    existing = current
    existing_2 = current_2

    Y_trunc = (np.array(Y))[0:n_tsteps]
    Z_trunc = (np.array(Z))[0:n_tsteps]
    X_trunc = (np.array(X))[0:n_tsteps]


    ### pi update (eqn 10)
    pi = (np.sum(Y_trunc, axis=0) / (np.sum(Y_trunc, axis=0) + np.sum(Z_trunc, axis=0)))
    pi_in = pi.copy()
    for i in range(lim):
        pi_in[i] = (1 - pi[i])

    ### Calculate theta matrix (not time dependent) (eqn 2)
    for i in range(lim):
        sum = 0
        for j in range(lim):
            if j != i:
                sum += s[j] * np.exp(-beta * dist[i, j])
        for j in range(lim):
            if j == i:
                theta[i, j] = 1 - pi[i]
            else:
                theta[i, j] = pi[i] * (s[j] * np.exp(-beta * dist[i, j])) / (sum)

    ### Calculate mu given theta (assumes no equal distances)
    for i in range(lim):
        for j in range(lim):
            sum = 0
            for t in range(n_tsteps):
                sum += N[t][j]
            mu[i, j] = sum * theta[j][i]

    pi_ext = pi[np.newaxis, :]
    pi_in_ext = pi_in[np.newaxis, :]
    mu_ext = mu[np.newaxis, :]
    ones = np.ones(X_trunc.shape)
    log_mu_ext_ones = np.log(mu_ext) * ones
    log_N_pi = np.log(N_trunc * pi_ext)
    log_N_pi_in = np.log(N_trunc * pi_in_ext)


    ### flattening and concatenating X, Y, Z to feed into optimisation
    dx = np.array(X).shape
    dy = np.array(Y).shape
    dz = np.array(Z).shape

    Xarr = np.array(X).flatten()
    Y_plus = Xarr.shape[0]
    Yarr = np.array(Y).flatten()
    Z_plus = Yarr.shape[0] + Xarr.shape[0]
    Zarr = np.array(Z).flatten()

    comb = np.concatenate((Xarr, Yarr, Zarr,))

    bnds = []
    for i in range(comb.shape[0]):
        bnds.append((0.0001, None))

    newvals = scipy.optimize.fmin_l_bfgs_b(loglik, comb, args=(mu, pi, N_trunc), fprime=jacob, bounds=bnds, epsilon=1e-8, maxiter=1000, factr=nv_factr, pgtol=nv_pgtol)
    XYZ = newvals[0]
    current = newvals[1]
    print '---------------------------'
    try:
        assert newvals[2]['warnflag'] == 0
    except AssertionError as err:
        print("XYZ error ", newvals[2]['task'])
        print(err)

    # newvals = scipy.optimize.minimize(loglik, comb, args=(mu, pi, N_trunc), jac=jacob, method="L-BFGS-B", bounds=bnds)
    # print '---------------------------'
    # try:
    #     assert newvals.success
    # except AssertionError as err:
    #     print("XYZ error ", newvals.message)
    #     print(err)
    # XYZ = newvals.x
    # current = newvals.fun


    print 'Current approx Log Likelihood = ', current
    app_list.append(current)
    for i in XYZ:
        if i < 0:
            print "XYZ bounds exceeded"

    ### Putting X, Y, Z back into original format
    for t in range(n_tsteps):
        for i in range(lim):
            Y[t][i] = XYZ[Y_plus + t * dy[1] + i]
            Z[t][i] = XYZ[Z_plus + t * dz[1] + i]
            for j in range(lim):
                X[t][i][j] = XYZ[t * dx[2] * dx[1] + i * dx[2] + j]

    ### Flattening and concatenating s and beta for optimisation
    sarr = np.array(s).flatten()
    beta = np.array([beta])
    sbeta = np.concatenate((sarr, beta))

    ### Maximising f_s_b
    bnds = []
    for i in range(sbeta.shape[0]-1):
        bnds.append((0, None))  ### s and beta bounds
    bnds.append((None, None))

    # newvals2 = scipy.optimize.minimize(f_s_b, sbeta, args=(Y_trunc, X_trunc), method="L-BFGS-B", bounds=bnds)
    # try:
    #     assert newvals2.success
    # except AssertionError as err:
    #     print("beta error ", newvals2.message)
    #     print(err)
    # newsbeta = newvals2.x
    # print 'Current f_s_b func val: ', newvals2.fun
    # f_list.append(newvals2.fun)
    # current_2 = newvals2.fun

    newvals2 = scipy.optimize.fmin_l_bfgs_b(f_s_b, sbeta, args=(Y_trunc, X_trunc), approx_grad=True, bounds=bnds, epsilon=1e-12, maxiter=1000, factr=beta_factr, pgtol=beta_pgtol)
    newsbeta = newvals2[0]
    print 'Current f_s_b func val: ', newvals2[1]
    f_list.append(newvals2[1])
    current_2 = newvals2[1]
    try:
        assert newvals2[2]['warnflag'] == 0
    except AssertionError as err:
        print("beta error ", newvals2[2]['task'])
        print(err)
        print "FORCING EXIT"
        exit()


    ### Putting s and beta back into original form
    for i in range(newsbeta.shape[0] - 1):
        s[i] = newsbeta[i]
    beta = newsbeta[-1]
    beta_list.append(beta)
    # print 's = ', s
    print 'beta = ', beta
    for sval in s:
        if sval < 0:
            print "f_s_b bounds exceeded"

    if abs((existing - current)/current)*100 < .1 and abs((existing_2 - current_2)/current_2)*100 < .1:
        print "Converged to within 0.1%"
        conv = True

stop1 = timeit.default_timer()

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
for i in range(lim):
    num = 0
    denom = 0
    for t in range(n_tsteps):
        num += Y[t][i]
        denom += Y[t][i] + Z[t][i]
    pi_f.append((num / denom))
pi_f = np.array(pi_f)

### Defining matrices with extra dimensions to avoid broadcasting problem in Numba nopython mode
ones = np.ones((n_tsteps, lim, lim))

pi_in_log = np.zeros((lim))
for h in range(lim):
    pi_in_log[h] = np.log(1 - pi_f[h])
pi_in_log = pi_in_log[np.newaxis, np.newaxis, :] * ones / ones

diag = np.zeros((lim, lim))
np.fill_diagonal(diag, 1)
diag = diag[np.newaxis, :, :] * ones / ones

o_diag = np.ones((lim, lim))
np.fill_diagonal(o_diag, 0)

o_diag_ext = o_diag[np.newaxis, :, :] * ones / ones

mult = np.log(pi_f[:, np.newaxis]) + np.log(np.array(s)[:, np.newaxis]) - beta * dist
mult = mult[np.newaxis, :, :] * ones / ones

mult2 = - np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-beta * dist) * o_diag, axis=1))
mult2 = mult2[np.newaxis, :, np.newaxis] * ones / ones

M_trun = np.zeros((n_tsteps, lim, lim))
M_trun_t = np.zeros((n_tsteps, lim, lim))

#######################################################

@numba.jit(parallel=True, nopython=True, nogil=True)
def jacob_2(Marr, beta, s, dist):
    M_der = M_trun.copy()
    M_sum_1 = np.zeros((n_tsteps, lim))
    M_sum_2 = np.zeros((n_tsteps, lim))
    logsum = np.zeros(lim)
    for t in prange(n_tsteps):
        for i in prange(lim):
            for j in prange(lim):
                M_sum_1[t][i] += Marr[t * dM[2] * dM[1] + i * dM[2] + j]
                M_sum_2[t][j] += Marr[t * dM[2] * dM[1] + i * dM[2] + j] ###Check
    for i in prange(lim):
        for k in prange(lim):
            logsum[i] += s[k] * np.exp(-beta * dist[i][k])
    for t in prange(n_tsteps):
        for i in prange(lim):
            for j in prange(lim):
                if i == j:
                    M_der[t][i][j] = np.log(1 - pi[i]) - np.log(Marr[t * dM[2] * dM[1] + i * dM[2] + j]) + lmbda * (N_trunc[t][i] + N_trunc_p1[t][j] - M_sum_1[t][i] - M_sum_2[t][j])
                else:
                    M_der[t][i][j] = np.log(pi[i]) + np.log(s[i]) - beta * dist[i][j] - np.log(logsum[i]) - np.log(Marr[t * dM[2] * dM[1] + i * dM[2] + j]) + lmbda * (N_trunc[t][i] + N_trunc_p1[t][j] - M_sum_1[t][i] - M_sum_2[t][j])
    M_der = M_der.flatten()
    return M_der

########################################################

### Exact log likelihood (eqn 4)
@numba.jit(parallel=True, nopython=True, nogil=True)
def loglik_ex(Marr, beta, s, dist):
    M_trunc = M_trun.copy()
    M_trunc_t = M_trun_t.copy()
    for t in prange(n_tsteps):
        for i in prange(lim):
            for j in prange(lim):
                M_trunc[t][i][j] = (Marr[t * dM[2] * dM[1] + i * dM[2] + j])
                M_trunc_t[t][j][i] = (Marr[t * dM[2] * dM[1] + i * dM[2] + j])

    fst = np.sum(np.square(np.sum(N_trunc - np.sum(M_trunc, axis=2), axis=1)))
    snd = np.sum(np.square(np.sum(N_trunc_p1 - np.sum(M_trunc_t, axis=2), axis=1)))

    first = pi_in_log * M_trunc * diag
    second = M_trunc * o_diag_ext * mult + M_trunc * o_diag_ext * mult2
    third = (M_trunc - M_trunc * np.log(M_trunc)) * o_diag_ext
    return -(np.sum(first + second + third) - (lmbda/2.) * (fst + snd))

### Performing final optimisation

bnds = []
for i in range(Marr.shape[0]):
    if i in ind_lst:
        bnds.append((0.1, 0.4))
    else:
        bnds.append((0.1, None))


Mvals = scipy.optimize.fmin_l_bfgs_b(loglik_ex, Marr, fprime=jacob_2, args=(beta, s, dist), bounds=bnds, epsilon=1e-8, maxiter=1000, factr=Mv_factr, pgtol=Mv_pgtol)
Mfinal = Mvals[0]
print '---------------------------'
try:
    assert Mvals[2]['warnflag'] == 0
except AssertionError as err:
    print("Mvals error ", Mvals[2]['task'])
    print(err)

print 'Final Log Likelihood value: ', Mvals[1]



## Returning M matrices to original form
for t in range(n_tsteps):
    for i in range(lim):
        for j in range(lim):
            M[t][i][j] = int(Mfinal[t * dM[2] * dM[1] + i * dM[2] + j])

stop = timeit.default_timer()


print 'Number of regions = ', lim
print 'Number of timesteps = ', n_tsteps
print 'beta = ', beta
print 's = ', s
print 'M(t=0) = \n', M[0]
print 'Run time loop = ', np.round(((stop1 - start)/60),2), 'mins'
print 'Run time final optimisation = ', np.round(((stop - stop1)/60),2), 'mins'

dev = 0
Tr = 0
for i in range(lim):
    for j in range(lim):
        dev += np.abs(M[0][i,j] - M_true[0][i,j])
        Tr += M_true[0][i,j]
avg_dev = dev/(lim * lim)
avg_tr = Tr/(lim * lim)

print 'Average deviation = ', np.round(avg_dev, 1)
print 'Average true value = ', np.round(avg_tr, 1)

num = 0
denom = 0
for i in range(n_tsteps):
    for j in range(lim):
        for k in range(lim):
            num += np.abs(M_true[i][j][k] - M[i][j][k])
            denom +=M_true[i][j][k]

NAE = num/denom


print "NAE = ", NAE


###################### END



