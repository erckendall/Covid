import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
import timeit
warnings.filterwarnings("ignore")
import numba




############ Note that in the final M optimisation, the only things that matter are the pi s and beta, not the X, Y, Z.
############### Hence, convergence should be based on convergence of these quantities, esp s and beta


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
lim = 8 #len(names)
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
for t in range(n_tsteps + 1):  # originally had  +1 here, but think by timesteps the authours actually mean time points
    N_t = []
    for val in ordered_vals_f_2020:
        N_t.append(val[t])
    N.append(N_t)

for t in range(n_tsteps):  # originally had  +1 here, but think by timesteps the authours actually mean time points
    for val in ordered_vals_f_2020:
        M_t = np.ones((len(names), len(names)))
    M.append(M_t * 10)
    Cut.append(K_cut)

### Initialise s for each region (not a function of time)
for i in range(len(names)):
    s.append(0.5)

### For each timestep, initialise list of X_t_i_j , Y_t_i and Z_t_i for each i in names
for t in range(n_tsteps):
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
@numba.jit(nopython = True, parallel = True, nogil=True)
def f_s_b(sbeta):

    Y_trun = Y_wt.copy()
    X_trun = X_wt.copy()
    distX = dist_X.copy()

    beta = sbeta[-1]
    s = []
    for i in range(lim):
        s.append(sbeta[i])
    s = np.array(s)
    np.expand_dims(s, axis=1)
    np.expand_dims(s, axis=0)
    ar = np.ones((lim, lim))
    np.fill_diagonal(ar, 0)
    fst = np.sum(X_trun * np.log(s))
    snd = -np.sum(np.sum(Y_trun, axis=0) * np.log(np.sum(s * np.exp(-beta * dist) * ar, axis=1)))
    thd = -beta * np.sum(distX)
    return -(fst+snd+thd)#, fst, snd, thd ### negative here as we want to maximise the positive value (minimise negative)



### Define approximate log likelihood function
@numba.jit(nopython = True, parallel = True, nogil=True)
def loglik(comb):

    Y_trunc = Y_wtf.copy()
    Z_trunc = Z_wtf.copy()
    X_trunc = X_wtf.copy()
    Xlog = np.log(X_trunc)
    Y_pl = Y_plus
    Z_pl = Z_plus

    N_trun = N_trunc.copy()
    N_trun_p1 = N_trunc_p1.copy()

    Xmu = Xmu1.copy()

    for t in range(n_tsteps):
        for i in range(lim):
            Y_trunc[t][i] = comb[Y_pl + t * dy[1] + i]
            Z_trunc[t][i] = comb[Z_pl + t * dz[1] + i]
            for j in range(lim):
                X_trunc[t][i][j] = comb[t * dx[2] * dx[1] + i * dx[2] + j]

    fst = np.sum(Y_trunc * np.log(N_trun * pi) + Y_trunc - Y_trunc * np.log(Y_trunc))
    snd = np.sum(Z_trunc * np.log(N_trun * pi_in) + Z_trunc - Z_trunc * np.log(Z_trunc))
    thd = np.sum(Xmu + X_trunc - X_trunc*Xlog)
    first = np.abs(np.sum((N_trun - Y_trunc - Z_trunc), axis=1))**2
    second = np.abs(np.sum((N_trun_p1 - np.sum(X_trunc, axis=2)), axis=1))**2 # took out Z_trunc after conversation with Lerh
    tot = np.sum(first + second)
    return -(fst + snd + thd - (lmbda/2) * tot)


##################### Begin while here:
# ### Maximise loglik - lmbda/2*penalty. Implement constraints X,Y,Z >= 0
# ### Update pi with new  X Y Z  using eq. 10
# ### Maximise f_s_b - update s and beta
# ### Update theta and thereforemu with new s and pi values

current_2 = 0.
current = 0.
conv = False


while conv == False:
    existing = current
    existing_2 = current_2

    Y_wtf = (np.array(Y))
    Z_wtf = (np.array(Z))
    X_wtf = (np.array(X))

    N_trunc = (np.array(N))[0:(n_tsteps)]
    N_trunc_p1 = (np.array(N))[1:(n_tsteps + 1)]

    ### pi update (eqn 10)
    pi = (np.sum(np.array(Y), axis=0) / (np.sum(np.array(Y), axis=0) + np.sum(np.array(Z), axis=0))).tolist()
    pi = np.array(pi)

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
            for t in range(n_tsteps):
                sum += N[t][j]
            mu[i, j] = sum * theta[j][i] ### Weird - what is t here

    Xmu1 = X_wtf * np.expand_dims(np.log(mu), axis=0)
    np.expand_dims(pi, axis=0)
    pi_in = np.ones(lim)
    for i in range(lim):
        pi_in[i] = 1 - pi[i]

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

    ### L-BFGS-B method doesn't respect bounds so need to modify
    bnds = []
    for i in range(comb.shape[0]):
        bnds.append((0, None))
    opt = {'maxiter': 2000, 'ftol': 10.}
    newvals = scipy.optimize.minimize(fun=loglik, x0=comb, method="SLSQP", bounds=bnds, options=opt, tol=10.)
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

    Y_wt = (np.array(Y).copy())
    Z_wt = (np.array(Z).copy())
    X_wt = (np.array(X).copy())
    print X_wt[0][0][0]
    dist_X = np.expand_dims(dist, axis=0) * X_wt


    ### Flattening and concatenating s and beta for optimisation
    sarr = np.array(s).flatten()
    beta = np.array([beta])
    sbeta = np.concatenate((sarr, beta))

    ### Maximising f_s_b
    ### Note: depending on whether fsb update before or after XYZ update there is different behaviour
    bnds = []
    for i in range(sbeta.shape[0]):
        bnds.append((0, None))  ### s and beta bounds

    # newvals = scipy.optimize.fmin_l_bfgs_b(f_s_b, sbeta, approx_grad=True, bounds=bnds, epsilon=1e-10, maxiter=20000, factr=1e7, pgtol=1e-3)
    newvals = scipy.optimize.fmin_l_bfgs_b(f_s_b, sbeta, approx_grad=True, bounds=bnds, epsilon=1e-10, maxiter=20000, pgtol=1)
    try:
        assert newvals[2]['warnflag'] == 0
    except AssertionError as err:
        print("beta error ", newvals[2]['task'])
        print(err)
    newsbeta = newvals[0]
    print 'f_s_b func val: ', newvals[1]
    current_2 = newvals[1]

    ### Putting s and beta back into original form
    for i in range(newsbeta.shape[0] - 1):
        s[i] = newsbeta[i]
    beta = newsbeta[-1]
    print 's = ', s
    print 'beta = ', beta
    for sval in s:
        if sval < 0:
            print "f_s_b bounds exceeded"

    if abs((existing - current)/current)*100 < .1 and abs((existing_2 - current_2)/current_2)*100 < .001:
        print "Converged to within 0.001%"
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
    for t in range(n_tsteps):
        num += Y[t][i]
        denom += Y[t][i] + Z[t][i]
    pi_f.append((num / denom))
pi_f = np.array(pi_f)
pi_in = np.zeros((lim))
for h in range(lim):
    pi_in[h] = (1 - pi_f[h])
pi_in = pi_in[np.newaxis, np.newaxis, :]

o_diag = np.ones((lim, lim,))
diag = np.zeros((lim, lim,))
np.fill_diagonal(o_diag, 0)
np.fill_diagonal(diag, 1)

M_trun = np.ones((n_tsteps, lim, lim))
M_trun_t = np.ones((n_tsteps, lim, lim))

pi_f = np.expand_dims(pi_f, axis=1)
s = np.array(s)
s = np.expand_dims(s, axis=1)

mult = np.log(pi_f) + np.log(s) - beta * dist
mult2 = - np.log(np.sum(s * np.exp(-beta * dist) * o_diag, axis=1))

log_pi_in_diag = np.squeeze(np.log(pi_in)*diag)

o_diag = o_diag[np.newaxis, :, :] * M_trun
log_pi_in_diag = log_pi_in_diag[np.newaxis, :, :] * M_trun
mult2 = mult2[np.newaxis,:,np.newaxis] * M_trun
mult = mult[np.newaxis, :, :] * M_trun


### Exact log likelihood (eqn 4) ### Need to figure out how K_cut can help here - currently just imposed in initial values, not strict rule
### Probably needs to go into the bounds in the optimisation
### Also needs to go into the XYZ bounds - check defs

@numba.jit(nopython=True, parallel=True, nogil=True)
def loglik_ex(Marr):

    N_trun = N_trunc.copy()
    N_trun_p1 = N_trunc_p1.copy()

    M_trunc = M_trun.copy()
    M_trunc_t = M_trun_t.copy()

    for t in range(n_tsteps):
        for i in range(lim):
            for j in range(lim):
                M_trunc[t][i][j] = (Marr[t * dM[2] * dM[1] + i * dM[2] + j])
                M_trunc_t[t][j][i] = (Marr[t * dM[2] * dM[1] + i * dM[2] + j])

    fst = np.sum(np.abs(np.sum(N_trun - np.sum(M_trunc, axis=2), axis=1))**2)
    snd = np.sum(np.abs(np.sum(N_trun_p1 - np.sum(M_trunc_t, axis=2), axis=1))**2)

    first = M_trunc * log_pi_in_diag
    second = M_trunc * o_diag * mult + M_trunc * o_diag * mult2
    third = (M_trunc - M_trunc * np.log(M_trunc))*o_diag
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
opt = {'maxiter': 1000, 'ftol': 2.3} ### Note that implementing this seems to severely change M output
Mvals = scipy.optimize.minimize(fun=loglik_ex, x0=Marr, method="SLSQP", bounds=bnds, constraints=cons, options=opt, tol=2.3)
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
print 's = ', s.tolist()
print 'M(t=0) = \n', M[0]
print 'Run time = ', ((stop - start)/60), 'mins'

###################### END




