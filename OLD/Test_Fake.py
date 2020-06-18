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
np.set_printoptions(formatter={'all':lambda x: str(int(x))})


lim = 64
run = 1000
lmbda = 1.
steps = 3
nv_factr = 1e9
nv_pgtol = .001
Mv_factr = 1e9
Mv_pgtol = .001


# For 36
# nv_factr = 1e8
# nv_pgtol = 1.
# Mv_factr = 1e7
# Mv_pgtol = .01
#NAE approx 0.37


# For 9
# lmbda = 10. (approx the same NAE for lambda = 1, but lmbda = 10 seems to look better)
# nv_factr = 1e5
# nv_pgtol = .01
# Mv_factr = 1e5
# Mv_pgtol = .01
#NAE approx 0.14


app_list = []
f_list = []
beta_list = []


N = np.load('Fake_64/N.npy')
M_true = np.load('Fake_64/M.npy')
dist = np.load('Fake_64/dist_fake.npy')

# print M_true[0]

### Want to include filter s.t. regions more than 80km cannot be accessed in one timestep
K_cut = np.where(dist > 1000, 0, 1)
# print K_cut


### Optional redefine number of timesteps e.g. only look at 12 hour section
n_tsteps = steps


###Initialise parameters. Note initial assumptions for M, beta, lmbda - evaluate sensitivity to these
start = timeit.default_timer()
Cut, M, X, Y, Z, pi, s = ([] for i in range(7))
theta, mu = (np.zeros((lim, lim)) for i in range(2))
beta = 0.

for t in range(n_tsteps):
    Cut.append(K_cut)

for t in range(n_tsteps):  # originally had  +1 here, but think by timesteps the authours actually mean time points
    M_t = np.ones((lim, lim))
    M.append(M_t * 1)


### Initialise s for each region (not a function of time)
for i in range(lim):
    s.append(.5)

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



### define f_s_b update (eq 11 to update s and beta) Need to maximise
@numba.jit(parallel=True)
def f_s_b(sbeta, Y_trunc, X_trunc):
    beta = sbeta[-1]
    s = sbeta[0:lim]
    ar = np.ones((lim, lim))
    np.fill_diagonal(ar, 0)
    fst = np.sum(X_trunc * np.log(np.array(s))[np.newaxis, :, np.newaxis])
    # snd = -np.sum(np.sum(Y_trunc, axis=0) * np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-1 * dist) * ar, axis=1)))
    snd = -np.sum(np.sum(Y_trunc, axis=0) * np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-beta * dist) * ar, axis=1)))
    thd = -beta * np.sum(dist[np.newaxis, :] * X_trunc)
    return -(fst+snd+thd)#, fst, snd, thd ### negative here as we want to maximise the positive value (minimise negative)


### Define approximate log likelihood function
@numba.jit(parallel=True, nopython=True, nogil=True, cache=True)
def loglik(comb, X_trunc, Y_trunc, Z_trunc):
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

current_2 = 0.
current = 0.
conv = False
N_trunc = (np.array(N))[0:(n_tsteps)]
N_trunc_p1 = (np.array(N))[1:(n_tsteps + 1)]

while conv == False:
    existing = current
    existing_2 = current_2

    Y_trunc = (np.array(Y))[0:(n_tsteps)]
    Z_trunc = (np.array(Z))[0:(n_tsteps)]
    X_trunc = (np.array(X))[0:(n_tsteps)]


    ### pi update (eqn 10)
    pi = (np.sum(Y_trunc, axis=0) / (np.sum(Y_trunc, axis=0) + np.sum(Z_trunc, axis=0))).tolist()

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
            mu[i, j] = sum * theta[j][i] ### Weird - what is t here
    pi_in = np.ones(lim)
    for i in range(lim):
        pi_in[i] = 1 - pi[i]

    pi_ext = np.array(pi)[np.newaxis, :]
    pi_in_ext = pi_in[np.newaxis, :]
    mu_ext = np.array(mu)[np.newaxis, :]
    ones = np.ones(X_trunc.shape)
    log_mu_ext_ones = np.log(mu_ext) * ones
    log_N_pi = np.log(N_trunc * pi_ext)
    log_N_pi_in = np.log(N_trunc * pi_in_ext)


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

    bnds = []
    for i in range(comb.shape[0]):
        bnds.append((0.1, None))

    newvals = scipy.optimize.fmin_l_bfgs_b(loglik, np.float64(comb), args=(X_trunc, Y_trunc, Z_trunc), approx_grad=True, bounds=bnds, epsilon=1e-8, maxiter=1000, factr=nv_factr, pgtol=nv_pgtol)
    XYZ = newvals[0]
    current = newvals[1]
    print '---------------------------'
    try:
        assert newvals[2]['warnflag'] == 0
    except AssertionError as err:
        print("XYZ error ", newvals[2]['task'])
        print(err)

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
    ### Note: depending on whether fsb update before or after XYZ update there is different behaviour
    bnds = []
    for i in range(sbeta.shape[0]-1):
        bnds.append((0, None))  ### s and beta bounds
    bnds.append((None, None))


    newvals = scipy.optimize.fmin_l_bfgs_b(f_s_b, sbeta, args=(Y_trunc, X_trunc), approx_grad=True, bounds=bnds, epsilon=1e-8, maxiter=1000, factr=1e7, pgtol=1.) #factr 1e7 for below lim 40
    try:
        assert newvals[2]['warnflag'] == 0
    except AssertionError as err:
        print("beta error ", newvals[2]['task'])
        print(err)
    newsbeta = newvals[0]
    print 'Current f_s_b func val: ', newvals[1]
    f_list.append(newvals[1])
    current_2 = newvals[1]

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


    # ar = np.ones((lim, lim))
    # np.fill_diagonal(ar, 0)
    # print -np.sum(np.sum(Y_trunc, axis=0) * np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-beta * dist) * ar, axis=1)))


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

### Exact log likelihood (eqn 4)
@numba.jit(parallel=True, nopython=True, nogil=True)
def loglik_ex(Marr):
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

##Need to scale to avoid abnormal termination in lnsrch





Mvals = scipy.optimize.fmin_l_bfgs_b(loglik_ex, Marr, approx_grad=True, bounds=bnds, epsilon=1e-8, maxiter=1000, factr=Mv_factr, pgtol=Mv_pgtol)
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





# ### Saving
#
# name = '{}{}{}{}{}{}'.format('day_', day, '_time_', tinit, '_run_', run)
#
# if not os.path.exists('{}{}'.format('Southland Complete Runs/', name)):
#     os.makedirs('{}{}'.format('Southland Complete Runs/', name))
#
# config = open('{}{}{}'.format('Southland Complete Runs/', name, '/config.txt'), "w+")
# config.write('{}{}{}'.format('n_regions = ', lim, '\n'))
# config.write('{}{}{}'.format('lambda = ', lmbda, '\n'))
# config.write('{}{}{}'.format('n_steps = ', steps, '\n'))
# config.write('{}{}{}'.format('nv_factr = ', nv_factr, '\n'))
# config.write('{}{}{}'.format('nv_pgtol = ', nv_pgtol, '\n'))
# config.write('{}{}{}'.format('Mv_factr = ', Mv_factr, '\n'))
# config.write('{}{}{}'.format('Mv_pgtol = ', Mv_pgtol, '\n'))
# config.write('{}{}{}'.format('Runtime (mins) = ', np.round(((stop - start)/60),2), '\n'))
#
#
# np.save('{}{}{}'.format('Southland Complete Runs/', name, '/app_LL.npy'), app_list)
# np.save('{}{}{}'.format('Southland Complete Runs/', name, '/fsb.npy'), f_list)
# np.save('{}{}{}'.format('Southland Complete Runs/', name, '/beta_list.npy'), beta_list)
# np.save('{}{}{}'.format('Southland Complete Runs/', name, '/s_final.npy'), s)
# np.save('{}{}{}'.format('Southland Complete Runs/', name, '/dist_mat.npy'), dist)
# np.save('{}{}{}'.format('Southland Complete Runs/', name, '/names.npy'), names)
#
# for t in range(n_tsteps):
#     np.save('{}{}{}{}{}'.format('Southland Complete Runs/', name, '/M_', t, '.npy'), M[t])