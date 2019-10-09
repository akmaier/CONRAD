import csv
import matplotlib.pyplot as plt
import numpy as np

fileerror = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/error.csv'

error = []

with open(fileerror) as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for index, in reader:
        error.append(index)



error = np.array(error)

error = error.astype(np.float)


# print(len(error))
x = np.linspace(1, len(error), len(error))

print(x)
print(error)
plt.plot(x, error, label="Error")
plt.legend()
plt.ylabel("Error")
plt.xlabel("Iteration")
# plt.axis([0, len(error), 0, 1])
plt.show()
# plt.savefig('C:/Users/Niklas/Documents/Uni/Bachelorarbeit/Bilder/BilderTestFilled/error.pdf')