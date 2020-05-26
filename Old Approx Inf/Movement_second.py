import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


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
names.pop(to_del[1]-1)
vals.pop(to_del[0])
vals.pop(to_del[1]-1)
dist = np.delete(dist, to_del[0], 0)
dist = np.delete(dist, (to_del[1]-1), 0)
dist = np.delete(dist, to_del[0], 1)
dist = np.delete(dist, (to_del[1]-1), 1)


### Looking at just Feb 2020 data:
count_j = 0
count_f = 0
for i in vals[0]:
    if int(i[0]) == 2020:
        if int(i[1]) == 2:
            count_f += 1
        if int(i[1]) == 1:
            count_j += 1
n_tsteps = count_f - 1 # -1 because number of timesteps one less than number of times ## Check if this is what paper means


### Ordering the data
ordered_vals_f_2020 = []
for val in vals:
    ordered = np.zeros(n_tsteps + 1)
    for i in val:
        if int(i[0]) == 2020 and int(i[1]) == 2:
            t_step = (int(i[2])-1)*24 + int(i[3])
            ordered[t_step] = int(i[4])
    ordered_vals_f_2020.append(ordered.tolist())



###############Checking weird midnight behaviour

lst_tot = []
for j in range(len(ordered_vals_f_2020[0])):
    if j % 24 == 0 and j < len(ordered_vals_f_2020[0]) - 1:
        fst = 0
        snd = 0
        th = 0
        frth = 0
        fth = 0
        for pl in range(len(names)):
            fst += ordered_vals_f_2020[pl][j-2]
            snd += ordered_vals_f_2020[pl][j-1]
            th += ordered_vals_f_2020[pl][j]
            frth += ordered_vals_f_2020[pl][j+1]
            fth += ordered_vals_f_2020[pl][j+2]
    lst_tot.append([fst, snd, th, frth, fth])
# times = ['22:00', '23:00', '00:00', '01:00', '02:00']
# for day in lst_tot:
    # plt.plot(times, day)
# plt.show()


### Optional redefine number of timesteps e.g. only look at 12 hour section
n_tsteps = 5


### Algorithm to estimate movements from static data (approximate inference)


###Initialise parameters. Note initial assumptions for M, beta, lmbda
N, M, X, Y, Z, pi, s = ([] for i in range(7))
theta, mu = (np.zeros((len(names), len(names))) for i in range(2))
beta = 1.
lmbda = 1.


### For each timestep we create an N_t list and an  M_t matrix
for t in range(n_tsteps+1):
    N_t = []
    for val in ordered_vals_f_2020:
        N_t.append(val[t])
        M_t = np.ones((len(names), len(names)))
    N.append(N_t)
    M.append(M_t)

### Initialise s for each region (not a function of time)
for i in range(len(names)):
    s.append(0.5)



### For each timestep, initialise list of X_t_i_j for each i in names: X[t][i][j]
for t in range(n_tsteps+1):
    X_t = []
    for i in range(len(names)):
        X_t_i_j = []
        for j in range(len(names)):
            X_t_i_j.append(M[t][j][i])
        X_t.append(X_t_i_j)
    X.append(X_t)


### For each timestep, initialise list of Y_t_i for each i in names: Y[t][i]
for t in range(n_tsteps+1):
    Y_t = []
    for i in range(len(names)):
        Y_t_i = 0
        for j in range(len(names)):
            if i != j:
                Y_t_i += M[t][i][j]
        Y_t.append(Y_t_i)
    Y.append(Y_t)


### For each timestep, initialise list of Z_t_i for each i in names: Z[t][i]
for t in range(n_tsteps+1):
    Z_t = []
    for i in range(len(names)):
        Z_t_i = M[t][i][i]
        Z_t.append(Y_t_i)
    Z.append(Y_t)


### pi update (eqn 10)
for i in range(len(names)):
    num = 0
    denom = 0
    for t in range(n_tsteps-2):
        num += Y[t][i]
        denom += Y[t][i] + Z[t][i]
    pi.append(num/denom)


### Calculate theta matrix (not time dependent) (eqn 2)
for i in range(len(names)):
    sum = 0
    for j in range(len(names)):
        if j != i:
            sum += s[j]* np.exp(-beta * dist[i,j])
    for j in range(len(names)):
        if j == i:
            theta[i,j] = 1 - pi[i]
        else:
            theta[i,j] = pi[i] * (s[j] * np.exp(-beta * dist[i,j]))/(sum)


### Calculate mu given theta (assumes no equal distances)
for i in range(len(names)):
    for j in range(len(names)):
        mu[i,j] = N[t][j] * theta[j][i]


# define f_s_b update (eq 11)
def f_s_b (s, beta):
    tot = 0
    for i in range(len(names)):
        first_third = 0
        second_pre = 0
        second = 0
        for t in range(n_tsteps - 2):
            second_pre += -Y[t][i]
        for j in range(len(names)):
            for t in range(n_tsteps - 2):
                first_third += (X[t][i][j]*np.log(s[i]) - beta*dist[i,j] * X[t][i][j])
            if j != i:
                second += s[j] * np.exp(-beta * dist[i, j])
        second_tot = second_pre * np.log(second)
        tot += first_third + second_tot
    return tot


### Surely the below penalty must include a sum over i even though not written in the paper
def penalty(x,y,z):
    tot = 0
    for t in range(n_tsteps-2):
        first = 0
        second = 0
        for i in range(len(names)):
            first += (N[t][i] - y[t][i] - z[t][i])
            second += (N[t+1][i] - z[t][i])
            for j in range(len(names)):
                second += -x[t][i][j]
        tot += (abs(first)**2 + abs(second)**2)
    return tot



### Define approximate log likelihood function
def loglik(x,y,z):
    tot = 0
    for t in range(n_tsteps-2):
        for i in range(len(names)):
            tot += (y[t][i] * np.log(N[t][i]*pi[i]) + y[t][i] - y[t][i]*np.log(y[t][i]))
            tot += (z[t][i] * np.log(N[t][i]*(1-pi[i])) + z[t][i] - z[t][i]*np.log(z[t][i]))
            for j in range(len(names)):
                tot += (x[t][i][j] * np.log(mu[i][j]) + x[t][i][j] - x[t][i][j]*np.log(x[t][i][j]))
    return tot


### Minimise this to maximise objective function
def to_min(x,y,z):
    return -(loglik(x,y,z) - (lmbda/2)*penalty(x,y,z))



# arr = np.array([X,Y,Z])
# print arr.shape
# scipy.optimize.minimize(fun=to_min, x0=arr, method='L-BFGS-B')




################# Note: all initialisations use append, can't use this for updating.........................
### Maximise loglik - lmbda/2*penalty. Implement constraints X,Y,Z >= 0
### Update pi with new  X Y Z  using eq. 10
### Maximise f_s_b - update s and beta
### Update theta and thereforemu with new s and pi values
### Iterate







