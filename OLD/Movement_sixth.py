import numpy as np
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
n_tsteps = 10

start = timeit.default_timer()
 ##################### Algorithm to estimate movements from static data (approximate inference)

###Initialise parameters. Note initial assumptions for M, beta, lmbda - evaluate sensitivity to these
N, M, X, Y, Z, pi, s = ([] for i in range(7))
theta, mu = (np.zeros((len(names), len(names))) for i in range(2))
beta = 0.
lmbda = 10.


### For each timestep we create an N_t list and an  M_t matrix
for t in range(n_tsteps + 1):
    N_t = []
    for val in ordered_vals_f_2020:
        N_t.append(val[t])
        M_t = np.ones((len(names), len(names)))
    N.append(N_t)
    M.append(M_t)

### Initialise s for each region (not a function of time)
for i in range(len(names)):
    s.append(0.5)


### For each timestep, initialise list of X_t_i_j , Y_t_i and Z_t_i for each i in names
for t in range(n_tsteps + 1):
    X_t, Y_t, Z_t = ([] for i in range(3))
    for i in range(len(names)):
        X_t_i_j = []
        Y_t_i = 0
        Z_t.append(M[t][i][i])
        for j in range(len(names)):
            X_t_i_j.append(M[t][j][i])
            if i != j:
                Y_t_i += M[t][i][j]
        X_t.append(X_t_i_j)
        Y_t.append(Y_t_i)
    X.append(X_t)
    Y.append(Y_t)
    Z.append(Z_t)


### Original f_s_b function
def orig_f_s_b(sbeta):
    tot = 0
    fst = 0
    snd = 0
    thd = 0
    for i in range(len(names)):
        first = 0
        third = 0
        second_pre = 0
        second = 0
        for t in range(n_tsteps - 2):
            second_pre += -Y[t][i]
        for j in range(len(names)):
            for t in range(n_tsteps - 2):
                first += X[t][i][j] * np.log(sbeta[i])
                third += (- sbeta[-1] * dist[i, j] * X[t][i][j])
            if j != i:
                second += sbeta[j] * np.exp(-sbeta[-1] * dist[i, j])
        second_tot = second_pre * np.log(second)
        tot += first + third + second_tot
        fst += first
        snd += second_tot
        thd += third
    return -tot#, fst, snd, thd  ### negative here as we want to maximise the positive value (minimise negative)


### define f_s_b update (eq 11 to update s and beta) Need to maximise
@numba.jit
def f_s_b(sbeta):
    beta = sbeta[-1]
    s = sbeta[0:lim]
    ar = np.ones((lim, lim))
    np.fill_diagonal(ar, 0)
    fst = np.sum(X_trunc * np.log(np.array(s))[np.newaxis, :, np.newaxis])
    snd = -np.sum(np.sum(Y_trunc, axis=0) * np.log(np.sum(np.array(s)[np.newaxis, :] * np.exp(-beta * dist) * ar, axis=1)))
    thd = -beta * np.sum(dist[np.newaxis, :] * X_trunc)
    return -(fst+snd+thd)#, fst, snd, thd ### negative here as we want to maximise the positive value (minimise negative)


### Surely the below penalty must include a sum over i even though not written in the paper
def penalty(comb):
    tot = 0
    for t in range(n_tsteps - 2):
        first = 0
        second = 0
        for i in range(len(names)):
            first += (N[t][i] - comb[Y_plus + t * dy[1] + i] - comb[Z_plus + t * dz[1] + i])
            second += (N[t + 1][i] - comb[Z_plus + t * dz[1] + i])
            for j in range(len(names)):
                second += -comb[t * dx[2] * dx[1] + i * dx[2] + j]
        tot += (abs(first) ** 2 + abs(second) ** 2)
    return tot


### Define approximate log likelihood function
@numba.jit
def loglik(comb):
    tot = 0
    for t in range(n_tsteps - 2):
        for i in range(len(names)):
            tot += (comb[Y_plus + t * dy[1] + i] * np.log(N[t][i] * pi[i]) + comb[Y_plus + t * dy[1] + i] - comb[
                Y_plus + t * dy[1] + i] * np.log(comb[Y_plus + t * dy[1] + i]))
            tot += (comb[Z_plus + t * dz[1] + i] * np.log(N[t][i] * (1 - pi[i])) + comb[Z_plus + t * dz[1] + i] - comb[
                Z_plus + t * dz[1] + i] * np.log(comb[Z_plus + t * dz[1] + i]))
            for j in range(len(names)):
                tot += (comb[t * dx[2] * dx[1] + i * dx[2] + j] * np.log(mu[i][j]) + comb[
                    t * dx[2] * dx[1] + i * dx[2] + j] - comb[t * dx[2] * dx[1] + i * dx[2] + j] * np.log(
                    comb[t * dx[2] * dx[1] + i * dx[2] + j]))
    return tot


### Minimise this to maximise objective function
def to_min(comb):
    return -(loglik(comb) - (lmbda / 2) * penalty(comb))



 ##################### Begin while here:
# ### Maximise loglik - lmbda/2*penalty. Implement constraints X,Y,Z >= 0
# ### Update pi with new  X Y Z  using eq. 10
# ### Maximise f_s_b - update s and beta
# ### Update theta and thereforemu with new s and pi values

current = -1000.
conv = False

while conv == False:
    existing = current

    Y_trunc = (np.array(Y))[0:(n_tsteps-2)]
    Z_trunc = (np.array(Z))[0:(n_tsteps - 2)]
    X_trunc = (np.array(X))[0:(n_tsteps - 2)]

    ### pi update (eqn 10)
    pi = (np.sum(Y_trunc, axis=0) / (np.sum(Y_trunc, axis=0) + np.sum(Z_trunc, axis=0))).tolist()

    ### Flattening and concatenating s and beta for optimisation
    sarr = np.array(s).flatten()
    beta = np.array([beta])
    sbeta = np.concatenate((sarr,beta))

    ### Maximising f_s_b
    ### Note: depending on whether fsb update before or after XYZ update there is different behaviour
    bnds = []
    for i in range(sbeta.shape[0]-1):
        bnds.append((0, None)) ### s bounds
    bnds.append((0, None)) ### beta bounds
    newvals = scipy.optimize.fmin_l_bfgs_b(f_s_b, sbeta,  approx_grad=True, bounds=bnds, epsilon=1e-8)
    try:
        assert newvals[2]['warnflag'] == 0
    except AssertionError as err:
        print("beta error ", newvals[2]['task'])
        print(err)
    newsbeta = newvals[0]


    ### Putting s and beta back into original form
    for i in range(newsbeta.shape[0]-1):
        s[i] = newsbeta[i]
    beta = newsbeta[-1]
    # print 's = ', s
    # print 'beta = ', beta
    for sval in s:
        if sval < 0:
            print "f_s_b bounds exceeded"

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
            mu[i, j] = N[t][j] * theta[j][i] ### Weird - what is t here

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
    newvals = scipy.optimize.minimize(fun=to_min, x0=comb, method="SLSQP", bounds=bnds)
    XYZ = newvals.x
    current = newvals.fun
    print 'Current = ', current
    for i in XYZ:
        if i < 0:
            print "XYZ bounds exceeded"
    if abs((existing - current)/current)*100 < 1:
        print "Converged to within 1%"
        conv = True

    ### Putting X, Y, Z back into original format
    for t in range(n_tsteps):
        for i in range(len(names)):
            Y[t][i] = XYZ[Y_plus + t * dy[1] + i]
            Z[t][i] = XYZ[Z_plus + t * dz[1] + i]
            for j in range(len(names)):
                X[t][i][j] = XYZ[t * dx[2] * dx[1] + i * dx[2] + j]

 ##################### End while


### Final pi update (eqn 10) - can do final s and beta as well but likely converged
for i in range(len(names)):
    num = 0
    denom = 0
    for t in range(n_tsteps - 2):
        num += Y[t][i]
        denom += Y[t][i] + Z[t][i]
    pi.append(num / denom)

 ##################### Final calculation of M

### Flattening for optimisation
Marr = np.array(M)
dM = Marr.shape
Marr = Marr.flatten()

### Surely the below penalty must include a sum over i even though not written in the paper
def orig_penalty(Marr):
    tot = 0
    for t in range(n_tsteps - 2):
        first = 0
        second = 0
        for i in range(len(names)):
            first += N[t][i]
            second += N[t + 1][i]
            for j in range(len(names)):
                first += (- Marr[t * dM[2] * dM[1] + i * dM[2] + j])
                second += (- Marr[t * dM[2] * dM[1] + j * dM[2] + i])
        tot += (abs(first) ** 2 + abs(second) ** 2)
    return tot

### Exact log likelihood (eqn 4)
@numba.jit()
def loglik_ex(Marr):
    first, second, third = (0,0,0)
    for t in range(n_tsteps-2):
        for i in range(len(names)):
            first += np.log(1 - pi[i]) * Marr[t * dM[2] * dM[1] + i * dM[2] + i]
    for t in range(n_tsteps-2):
        for i in range(len(names)):
            for j in range(len(names)):
                if j != i:
                    sum = np.log(pi[i]) + np.log(s[j]) - beta * dist[i,j]
                    s_sum = 0
                    for k in range(len(names)):
                        if k != i:
                            s_sum += s[k] * np.exp(-beta * dist[i,k])
                    sum += (-np.log(s_sum))
                    second += sum * Marr[t * dM[2] * dM[1] + i * dM[2] + j]
    for t in range(n_tsteps-2):
        for i in range(len(names)):
            for j in range(len(names)):
                if j != i:
                    third += (Marr[t * dM[2] * dM[1] + i * dM[2] + j] - Marr[t * dM[2] * dM[1] + i * dM[2] + j] * np.log(Marr[t * dM[2] * dM[1] + i * dM[2] + j]))
    return (first + second + third)


### Minimise this to maximise objective function for final M calculation
def to_min_final(Marr):
    return -(loglik_ex(Marr) - (lmbda / 2) * orig_penalty(Marr))

### Performing final optimisation
bnds = []
for i in range(Marr.shape[0]):
    bnds.append((0, None))
Mvals = scipy.optimize.minimize(fun=to_min_final, x0=Marr, method="SLSQP", bounds=bnds)
Mfinal = Mvals.x


### Returning M matrices to original form
for t in range(n_tsteps):
    for i in range(len(names)):
        for j in range(len(names)):
            M[t][i][j] = int(Mfinal[t * dM[2] * dM[1] + i * dM[2] + j])

stop = timeit.default_timer()

print '---------------------------'
print 'Number of regions = ', lim
print 'Number of timesteps = ', n_tsteps
print 'beta = ', beta
print 's = ', s
print 'M(t=0) = ', M[0]
print 'Run time = ', ((stop - start)/60), 'mins'


 ###################### END




# Current =  -98288.4066870347
# [0.8487811556097449, 0.7464427146320216, 0.6728807615299884, 0.9602279825682367, 0.6246869596301938]


