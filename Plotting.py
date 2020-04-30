import numpy as np
import matplotlib.pyplot as plt

Fland = np.load('Fiordland_Feb_19_20.npy')
Gore = np.load('Gore_Central_Feb_19_20.npy')
Inland = np.load('Inland_Water_Lake_Manapouri_Feb_19_20.npy')
Inver = np.load('Invercargill_Central_Feb_19_20.npy')
Oceanic = np.load('Oceanic_Southland_Region_Feb_19_20.npy')


date = []
for i in range(28):
    date.append(i+1)


plt.figure(figsize=(20,7))
plt.xlim(1,28)
plt.ylim(0.75, 2.8)
plt.plot(date, Fland[0], color='g', linestyle='-', label='Fiordland 2019')
plt.plot(date, Fland[1], color='g', linestyle='--', label='Fiordland 2020')
plt.plot(date, Gore[0], color='r', linestyle='-', label='Gore Central 2019')
plt.plot(date, Gore[1], color='r', linestyle='--', label='Gore Central 2020')
plt.plot(date, Inland[0], color='y', linestyle='-', label='Lake Manapouri 2019')
plt.plot(date, Inland[1], color='y', linestyle='--', label='Lake Manapouri 2020')
plt.plot(date, Inver[0], color='b', linestyle='-', label='Invercargill Central 2019')
plt.plot(date, Inver[1], color='b', linestyle='--', label='Invercargill Central 2020')
plt.plot(date, Oceanic[0], color='k', linestyle='-', label='Oceanic Southland 2019')
plt.plot(date, Oceanic[1], color='k', linestyle='--', label='Oceanic Southland 2020')
plt.ylabel('Midday Population / Midnight Population')
plt.xlabel('Date')
plt.title('Southland Region Population Movement: February 2019/2020')
plt.legend(frameon=False, ncol=5)
plt.show()