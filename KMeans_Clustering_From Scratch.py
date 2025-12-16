import numpy as np 
import matplotlib.pyplot as plt

class kmeansclustering():
    def __init__(self, k=3):
        self.k=k
        self.centroid=None

    @staticmethod
    def euclideandistance(data_point, centroid):
        return np.sqrt(np.sum((centroid-data_point)**2, axis=1)) 
    
    def fit(self, X, max_iterations):
        self.centroid= np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))
        
        for _ in range(max_iterations):
            y=[]
            for data_point in X:
                distance= kmeansclustering.euclideandistance(data_point, self.centroid)
                cluster_num=np.argmin(distance)             #we assume a cluster no to it
                y.append(cluster_num)
            y=np.array(y)
            cluster_centers=[]
            for i in range(self.k):
                cluster_indices = np.argwhere(y==i).flatten()   
                #this is the trick to convert a 2D array into a 1D array for easy calling at X
        
                if len(cluster_indices)==0:
                    cluster_centers.append(self.centroid[i])
                else:    
                    cluster_centers.append(np.mean(X[cluster_indices], axis=0)  )         
                    # remember y is the location of the term in x that belongs to the center                
            if np.max(np.abs(self.centroid - np.array(cluster_centers))) < 0.0001:
                break
            else:
                self.centroid = np.array((cluster_centers))
        return y

random_points = np.random.randint(0,100, (100, 2))

kmeans= kmeansclustering(k=3)
labels = kmeans.fit(random_points, max_iterations=200)

plt.scatter(random_points[:,0], random_points[:,1], c=labels)
plt.scatter(kmeans.centroid[:,0], kmeans.centroid[:,1], c=range(len(kmeans.centroid)), marker = "*", s=200)
plt.show()
