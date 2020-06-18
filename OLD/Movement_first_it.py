import numpy as np


names = np.load('southland_names.npy').tolist()
vals = np.load('southland_vals.npy').tolist()
dist = np.load('southland_distance_matrix.npy')


#### Note: Two localities have less data than the rest (Inlets Fiordland and Inlets Water Lake Hauroko)
#### As previously calculated, these two regions average less than 10 people, so we will ignore both.


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



####### Note also, seems only to be data for January and February - checked the .dat file from the telcos. Verified below:

# count_j = 0
# count_f = 0
# for i in vals[0]:
#     if int(i[1]) == 2:
#         count_f += 1
#     if int(i[1]) == 1:
#         count_j += 1
# print count_f, 2*28*24+24 # add 24 for leap year
# print count_j, 2*31*24



######### Looking at just Feb 2020 data:
count_j = 0
count_f = 0
for i in vals[0]:
    if int(i[0]) == 2020:
        if int(i[1]) == 2:
            count_f += 1
        if int(i[1]) == 1:
            count_j += 1
n_tsteps = count_f - 1 # -1 because number of timesteps one less than number of times


###### Note, need to order data, not necessarily written hour by hour
ordered_vals_f_2020 = []
for val in vals:
    ordered = np.zeros(n_tsteps + 1)
    for i in val:
        if int(i[0]) == 2020 and int(i[1]) == 2:
            t_step = (int(i[2])-1)*24 + int(i[3])
            ordered[t_step] = int(i[4])
    ordered_vals_f_2020.append(ordered.tolist())
# print ordered_vals_f_2020[0]



######### Algorithm

N = []
M = []
X = []
Y = []
Z = []
pi = []
s = [] #  do we have to define this first or calculate with first calculation of f_s_b
beta = 1. ## What would be an appropriate size for this value??
lmbda = 1.## What would be an appropriate value??
theta = np.zeros((len(names), len(names)))
mu = np.zeros((len(names), len(names)))
# delta = 80. ## (assumption that can't travel further than 80km in 1 hour)

for t in range(n_tsteps+1):
    N_t = []
    for val in ordered_vals_f_2020:
        N_t.append(val[t])
        M_t = np.ones((len(ordered_vals_f_2020), len(ordered_vals_f_2020))) # Initial value???
    N.append(N_t)
    M.append(M_t)

for i in range(len(names)):
    s.append(0.5)


###### X should be of order 3, i.e. delta is an index, can therefore be summed over in f_s_b why don't we just index it i,j since delta = f(i,j)
##### If not on a regular grid, may only be one i,j pair with a given distance, so the sum in the definition of X is trivial???
#### Gamma_i is always the whole grid, penalty is what encompasses preclusion of superluminal travel
for t in range(n_tsteps+1):
    X_t = []
    for i in range(len(names)):
        X_t_i = []
        for j in range(len(names)):
            # if dist[i,j] < delta: #### Is this supposed to be outer boundary or does the distance have to be equal to delta? Should be equal to
            X_t_i.append(M[t][j][i])
        X_t.append(X_t_i)
    X.append(X_t)


for t in range(n_tsteps+1):
    Y_t = []
    for i in range(len(names)):
        Y_t_i = 0
        for j in range(len(names)):
            if i != j:
                Y_t_i += M[t][i][j]
        Y_t.append(Y_t_i)
    Y.append(Y_t)


for t in range(n_tsteps+1):
    Z_t = []
    for i in range(len(names)):
        Z_t_i = M[t][i][i]
        Z_t.append(Y_t_i)
    Z.append(Y_t)


##### pi update (eqn 10)
for i in range(len(names)):
    num = 0
    denom = 0
    for t in range(n_tsteps-2):
        num += Y[t][i]
        denom += Y[t][i] + Z[t][i]
    pi.append(num/denom)


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

for i in range(len(names)):
    for j in range(len(names)):
        mu[i,j] = N[t][j] * theta[j][i]

# need to define app log_lik f_s_b update (eq 11)
## log_lik contains mu which  is calculated using theta_i_j,
# theta is calculated by equation 2
# What does the sum over delta mean in equation 11? or are we trying to optimise delta as well? Thought it was a fixed param

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
        tot +=first_third + second_tot
    return tot

#### Surely the below must include a sum over i even though not written in the  paper
def penalty(X,Y,Z):
    tot = 0
    for t in range(n_tsteps-2):
        first = 0
        second = 0
        for i in range(len(names)):
            first += (N[t][i] - Y[t][i] - Z[t][i])
            second += (N[t+1][i] - Z[t][i])
            for j in range(len(names)):
                second += -X[t][i][j]
        tot += (abs(first)**2 + abs(second)**2)
    return tot



def loglik(X, Y, Z):
    tot = 0
    for t in range(n_tsteps-2):
        for i in range(len(names)):
            tot += (Y[t][i] * np.log(N[t][i]*pi[i]) + Y[t][i] - Y[t][i]*np.log(Y[t][i]))
            tot += (Z[t][i] * np.log(N[t][i]*(1-pi[i])) + Z[t][i] - Z[t][i]*np.log(Z[t][i]))
            for j in range(len(names)):
                tot += (X[t][i][j] * np.log(mu[i][j]) + X[t][i][j] - X[t][i][j]*np.log(X[t][i][j]))
    return tot


### Global optimum via gradient based methods: should maximise loglik - lmbda/2*penalty

# contstaints X Y Z >= 0

########### Is the penalty term supposed to also sum over i???? Must be!  also, is the Gamma here supposed to also be Gamma_delta?
############ Penalty term not present in the approximate log likelihood function
# C_of_M = 0




