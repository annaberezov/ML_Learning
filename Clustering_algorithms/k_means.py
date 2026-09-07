import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Data Prep
## generate random data points
x,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)

## plot data points
fig = plt.figure(0)
plt.grid(True)
plt.scatter(x[:,0],x[:,1])
plt.show()

## scale the data points
scaler = StandardScaler()
x = scaler.fit_transform(x)

# K-Means Algorithm
## Initialize the centroids
## k is here set to be the same as the centers defined in the blobs. When working with real data, can use the elbow method
k = 3

clusters = {}
np.random.seed(23)

## creating the clusters with random centers
for i in range(k):
    center = 2*(2*np.random.random((x.shape[1],))-1)
    points = []
    cluster = {
        'center': center,
        'points':[]
        }
    clusters[i] = cluster

print(clusters)

## plotting the clusters with their centers
plt.scatter(x[:,0],x[:,1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker='*',c = 'red')
plt.show()

## Defining the distance
def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

## assigning the clusters
def assignClusters(x,clusters):
    for i in range(x.shape[0]): #x.shape[0] is basically the same as the length as it gives the number of rows
        distances = []
        for j in clusters:
            center = clusters[j]['center']
            d = distance(x[i],center)
            distances.append(d)
        clusterIndex = np.argmin(distances)
        clusters[clusterIndex]['points'].append(x[i])
    return clusters

## update the clusters
def updateClusters(clusters):
    for i in clusters:
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            newCenter = points.mean(axis=0)
            clusters[i]['center'] = newCenter
            clusters[i]['points'] = []
    return clusters

## predicting the cluster for the data points
def predClusters(x,clusters):
    pred = []
    for i in range(x.shape[0]):
        distances = []
        for j in clusters:
            distances.append(distance(x[i],clusters[j]['center']))
        pred.append(np.argmin(distances))
    return pred

# Execute the algorithm until convergence or max iterations are reached
maxIterations = 100

for _ in range(maxIterations):
    oldCenters = [clusters[i]['center'].copy() for i in clusters]

    clusters = assignClusters(x,clusters)
    clusters = updateClusters(clusters)

    newCenters = [clusters[i]['center'] for i in clusters]

    if np.allclose(oldCenters,newCenters):
        break

pred = predClusters(x,clusters)

# plotting the final clusters with their centers
plt.scatter(x[:,0],x[:,1],c=pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker='*',c = 'red')
plt.show()


## Can also use the sklearn library to implement the k-means algorithm
kmeans = KMeans(n_clusters=k)
kmeans.fit(x,y)
pred = kmeans.predict(x)

plt.scatter(x[:,0],x[:,1],c=pred)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',c = 'black')
plt.show()