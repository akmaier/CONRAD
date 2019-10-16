import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



filemean = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/mean.csv'
filehisto = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/values.csv'
filecount = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/count.csv'
nr_ell = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/nr_ell.csv'
threshs = 'C:/Reconstruction/CONRAD/src/edu/stanford/rsl/BA_Niklas/DataSheets/thresholds.csv'



histo = []
count = []
nr_ellipses = []


with open(filehisto) as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for index, in reader:
        histo.append(index)

with open(filecount) as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for index, in reader:
        count.append(index)

with open(nr_ell) as csv_file:
    reader = csv.reader(csv_file, delimiter=';')

    for index, in reader:
        nr_ellipses.append(index)
#
#

newhisto = [float(i) for i in histo]
newcount = [float(i) for i in count]





#scaling
scalehisto = max(newhisto) - min(newhisto)
scalecount = max(newcount) - min(newcount)

scale = (scalecount / scalehisto) *2

newhisto = [i * scale for i in newhisto]

histopoints = []
for i in range(0, len(newhisto)):
    point = [newhisto[i], newcount[i]]
    histopoints.append(point)
histopoints = np.array(histopoints)



# create kmeans object
kmeans = KMeans(n_clusters= int(float(nr_ellipses[0])))
# fit kmeans object to data
kmeans.fit(histopoints)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(histopoints)



centers = kmeans.cluster_centers_
means = [0]


for i in range(0, len(centers)):
    means.append(centers[i][0])
means.sort()

thresholds = []
for i in range(0, len(means)):
    if len(means)-1 == i:
        break
    threshold = (means[i] + means[i+1]) / 2
    thresholds.append(threshold)

thresholds = [i / (scale) for i in thresholds]
print(thresholds)


with open(threshs, "w") as csv_file:
    spamwriter = csv.writer(csv_file, delimiter=':')
    spamwriter.writerows(map(lambda x: [x], thresholds))


plt.plot(newhisto, newcount, "ro", label="histogram")
plt.legend()
plt.ylabel("count")
plt.xlabel("value")
plt.show()

plt.scatter(histopoints[y_km ==0,0], histopoints[y_km == 0,1], s=50, c='red')
plt.scatter(histopoints[y_km ==1,0], histopoints[y_km == 1,1], s=50, c='black')
plt.scatter(histopoints[y_km ==2,0], histopoints[y_km == 2,1], s=50, c='blue')
plt.scatter(histopoints[y_km ==3,0], histopoints[y_km == 3,1], s=50, c='cyan')
plt.show()
