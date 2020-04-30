import numpy as np
import matplotlib.pyplot as plt

data = np.load('southland_data.npy').tolist() # year, month, day, hour, number, placename


name_list = []
name = 'start'
for i in data:
    if i[5] != name and i[5] not in name_list:
        name_list.append(i[5])
        name = i[5]
ls = []
for i in name_list:
    ls.append([])
ind = -1
for i in name_list:
    ind += 1
    for d in data:
        if d[5] == i:
            ls[ind].append(d[:-1])
# np.save('southland_names.npy', name_list)
# np.save('southland_vals.npy', ls)



### Calculate distance matrix between Southland locations
lat_long = np.load('southland_lat_long.npy')
dist_mat = np.zeros((len(name_list),len(name_list)))
r = 6371  # Average radius of earth
for i in range(len(name_list)):
    for j in range(len(name_list)):
        p1 = lat_long[i][2][0] # lat (radians)
        l1 = lat_long[i][2][1]  # long (radians)
        p2 = lat_long[j][2][0] # lat (radians)
        l2 = lat_long[j][2][1]  # long (radians)
        h = np.sin((p1-p2)/2)**2 + np.cos(p1) * np.cos(p2) * np.sin((l1-l2)/2)**2
        dist_mat[i,j] = 2 * r * np.arcsin(np.sqrt(h))
# np.save('Southland_distance_matrix.npy', dist_mat)






### Calculate daily changes midnight to midday for specific subregions 0, 4, 31, 40, 64

# feb_19_20 = []
# for year in range(2019, 2021):
#     reg_ind = 64
#     test = ls[reg_ind]  ## i.e. this for just one subregion
#     days = []
#     nights = []
#     for i in range(29):
#         days.append([])
#         nights.append([])
#     for i in test:
#         for j in range(28):
#             if int(i[0]) == year and int(i[1]) == 2 and int(i[2]) == j+1:
#                 if int(i[3]) == 12:
#                     days[j+1].append(i[4])
#                 if int(i[3]) == 0:
#                     nights[j+1].append(i[4])
#     days.pop(0)
#     nights.pop(0)
#     delta = []
#     for i in range(len(days)):
#         delta.append(float(days[i][0])/float(nights[i][0]))
#     feb_19_20.append(delta)
# np.save('Oceanic_Southland_Region_Feb_19_20.npy', feb_19_20) ## Change as appropriate






# # Calculate ratio of standard deviation to average for 2019

# rat = []
# for i in range(len(name_list)):
#     vals = [0]
#     for j in ls[i]:
#         if int(j[0]) == 2019:
#             vals.append(int(j[4]))
#     avg = np.average(vals)
#     sd = np.std(vals)
#     if avg > 10:
#         rat.append(sd/avg)
#     else:
#         rat.append(0)
#
# plt.figure(figsize=(20,6))
# plt.title('Southland Region 2019')
# plt.bar(name_list, rat)
# plt.ylabel('Standard deviation / average')
# plt.xticks(rotation=90)
# plt.gcf().subplots_adjust(bottom=0.4)
# plt.show()
