import csv
import numpy as np

code_name  = []
with open('sa2/sa2_simp_csv.csv', 'rt') as file:
    data = csv.reader(file)
    for row in data:
        code_name.append([row[0], row[1]])
code_name.pop(0)
# print code_name


southland_names = []
with open('braoder-regions/sa2_larger_csv.csv', 'rt') as file_lg:
    data_lg = csv.reader(file_lg)
    num = 0
    for row in data_lg:
        if num > 0:
            if row[3] == 'Southland Region':
                southland_names.append(row[1])
        num += 1
# print southland_names


southland_codes = []
for cn in code_name:
    if cn[1] in southland_names:
        southland_codes.append(int(cn[0]))
# print southland_codes




# southland_lat_long = []
# with open('Centroids/centroids.csv', 'rt') as cents:
#     data_cents = csv.reader(cents)
#     num = 0
#     for row in data_cents:
#         if num > 0:
#             if int(row[1]) in southland_codes:
#                 lat = (float(row[7])/180.)*np.pi
#                 long = (float(row[8])/180.)*np.pi
#                 southland_lat_long.append([int(row[1]), row[2], [lat, long]])
#         num += 1
# # print southland_lat_long
# np.save('southland_lat_long.npy', southland_lat_long)




south  = []
with open('data_covid.dat', 'rt') as telco:
    t_data = csv.reader(telco)
    num = 0
    for row in t_data:
        if num > 0:
            if int(row[2]) in southland_codes:
                for code in code_name:
                    if int(code[0]) == int(row[2]):
                        place = code[1]
                ints = [row[3][0], row[3][1], row[3][2], row[3][3]]
                st = [str(i) for i in ints]
                string = "".join(st)
                year = int(string)
                # print 'Year = ', year
                ints = [row[3][5], row[3][6]]
                st = [str(i) for i in ints]
                string = "".join(st)
                month = int(string)
                # print 'Month = ', month
                ints = [row[3][8], row[3][9]]
                st = [str(i) for i in ints]
                string = "".join(st)
                day = int(string)
                # print 'Day = ', day
                ints = [row[3][11], row[3][12]]
                st = [str(i) for i in ints]
                string = "".join(st)
                hour = int(string)
                # print 'Hour = ', hour
                peop = int(row[4])
                south.append([year, month, day, hour, peop, place])
        num += 1
        if num % 1000 == 0:
            print num

np.save('southland_data.npy', south)



