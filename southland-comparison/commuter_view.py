from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


import csv
with open('commuter_view_data.csv', newline='') as csvfile:
    data = csv.reader(csvfile, delimiter=',', quotechar='|')
    lst = []
    for row in data:
        lst.append([row[0], row[1], row[2], row[3]])
lst.pop(0)
lst_2 = []
ind = []
cnt = -1
for i in lst:
    if i[3] == '(Oreti Beach)':
        in_oreti = i[2]
        out_oreti = i[1]
for i in lst:
    cnt += 1
    if i[2] != '-' and i[3] != '(Oreti Beach)':
        lst_2.append(i)
        ind.append(cnt)
    if i[0] == 'Avenal':
        avenal_ind = cnt
    if i[0] == 'Gore Main':
        main_ind = cnt
for i in range(len(lst_2)):
    if lst_2[i][0] == 'Gladstone (Invercargill City)':
        gladstone_ind = i
    if lst_2[i][0] == 'Gore Central':
        gore_ind = i
################################Fix this: adding too much???

for i in range(len(lst_2)):
    if lst_2[i][0] == 'Otatara':
        ind_otatara = i
val_2 = float(lst_2[ind_otatara][2]) + float(in_oreti)
val_1 = float(lst_2[ind_otatara][1]) + float(out_oreti)
lst_2[ind_otatara][2] = val_2
lst_2[ind_otatara][1] = val_1


# Add Oreti beach value to Otatara in the commuter view version
# Oreti beach is not the same as the Oreti river
# Gore central + Gore Main
# Gladstone + Avenal

rnd_out = np.load("outs.npy")
rnd_in = np.load("ins.npy")
rnd_out_2 = []
rnd_in_2 = []
for i in range((rnd_out.shape[0])):
    if i in ind:
        rnd_out_2.append(float((rnd_out[i][1]).decode('utf-8')))
        rnd_in_2.append(float((rnd_in[i][1]).decode('utf-8')))
for i in range(len(rnd_out_2)):
    if i == gladstone_ind:
        rnd_out_2[i] += float((rnd_out[avenal_ind][1]).decode('utf-8'))
        rnd_in_2[i] += float((rnd_in[avenal_ind][1]).decode('utf-8'))
    if i == gore_ind:
        rnd_out_2[i] += float((rnd_out[main_ind][1]).decode('utf-8'))
        rnd_in_2[i] += float((rnd_in[main_ind][1]).decode('utf-8'))


names = []
in_inf = []
in_snz = []
out_inf = []
out_snz = []
xs = []
# name, commuter view value, inference value
for i in range(len(lst_2)):
    names.append(lst_2[i][0])
    out_snz.append([float(lst_2[i][1])])
    out_inf.append([rnd_out_2[i]])
    in_inf.append([rnd_in_2[i]])
    in_snz.append([float(lst_2[i][2])])
    xs.append(i)

plt.figure(figsize=(20, 10))
plt.title('Southland - Inbound')
plt.plot(in_snz, marker='o', label='SNZ CommuterView')
plt.plot(in_inf, marker='o', label='Inference Model')
# plt.title('Southland - Outbound')
# plt.plot(out_snz, marker='o', label='SNZ CommuterView')
# plt.plot(out_inf, marker='o', label='Inference Model')
plt.xticks(xs ,names, rotation=90)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
