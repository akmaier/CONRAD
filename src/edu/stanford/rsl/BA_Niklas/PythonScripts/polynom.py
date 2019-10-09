import csv
import matplotlib.pyplot as plt
import numpy as np



fileabso = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/abso.csv'
filedark = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/dark.csv'
filereg = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/reg.csv'

abso = []
dark = []
reg = []


with open(fileabso) as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for index, in reader:
        abso.append(index)

with open(filedark) as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for index, in reader:
        dark.append(index)

with open(filereg) as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for index, in reader:
        reg.append(index)




abso = list(dict.fromkeys(abso))
dark = list(dict.fromkeys(dark))

abso = np.array(abso)
dark = np.array(dark)
#
abso = abso.astype(np.float)
dark = dark.astype(np.float)


maximum = (max(abso))
minimum = min(len(abso), len(dark))

newdark = dark[0:minimum]
newabso = abso[0:minimum]





x = np.linspace(0, maximum, 100)
y = float(reg[3]) * x**3 + float(reg[2]) * x**2 + float(reg[1]) * x + float(reg[0])
# reg[3]) * x**3 + float(reg[2]) * x**2 +
# x1= [1, 2, 4, 7, 4]
# y1= [1, 5.2, 5.2, 4, 2]
plt.plot(newabso, newdark,  "bo", label="correlation-points", markersize=1)
plt.plot(x, y, "r-", label="polynom")
plt.legend()
plt.ylabel("darkfield")
plt.xlabel("absorption")
plt.show()
plt.savefig('C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/polynom.pdf')

# plt.plot(newabso, newdark,  "bo", label="correlation-points", markersize=1)
# plt.legend()
# plt.ylabel("darkfield")
# plt.xlabel("absorption")
# plt.show()
#
# plt.plot(x, y, "r-", label="polynom")
# plt.legend()
# plt.ylabel("darkfield")
# plt.xlabel("absorption")
# plt.show()