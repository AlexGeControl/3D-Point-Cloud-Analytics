#!/opt/conda/envs/03-clustering/bin/python

# 文件功能： 实现 Spectral Clustering 算法

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class SpectralClustering(object):
    """
    SpectralClustering 

    Parameters
    ----------
    n_clusters: int
        Number of clusters 

    Attributes
    ----------

    """
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, **kwargs):
        self.__K = n_clusters
        self.__labels = None

    def fit(self, data):
        """
        Estimate the K centroids

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        Returns
        ----------
        None

        """
        # TODO 01: implement SpectralClustering fit 
        from sklearn.neighbors import kneighbors_graph
        from sklearn.metrics import pairwise_distances
        from scipy.sparse import csgraph
        from scipy.sparse import linalg

        N, _ = data.shape

        # create affinity matrix -- kNN for connectivity:
        A = pairwise_distances(data)
        # TODO: use better gamma estimation
        gamma = np.var(A)/4
        A = np.exp(-A**2/(2*gamma**2))
        # get laplacian matrix:
        L = csgraph.laplacian(A, normed=True)
        # spectral decomposition:
        eigval, eigvec = np.linalg.eig(L)
        # get features:
        idx_k_smallest = np.where(eigval < np.partition(eigval, self.__K)[self.__K])
        features = np.hstack([eigvec[:, i] for i in idx_k_smallest])
        # cluster using KMeans++
        k_means = KMeans(init='k-means++', n_clusters=self.__K, tol=1e-6)
        k_means.fit(features)
        # get cluster ids:
        self.__labels = k_means.labels_

    def predict(self, data):
        """
        Get cluster labels

        Parameters
        ----------
        data: numpy.ndarray
            Testing set as N-by-D numpy.ndarray

        Returns
        ----------
        result: numpy.ndarray
            data labels as (N, ) numpy.ndarray

        """
        return np.copy(self.__labels)

def generate_dataset(N=300, noise=0.07, random_state=42, visualize=False):
    """
    Generate dataset for spectral clustering

    Parameters
    ----------
    visualize: boolean
        Whether to visualize the generated data

    """
    from sklearn.datasets import make_moons
    
    X, y = make_moons(N, noise=noise, random_state=random_state)

    if visualize:
        fig, ax = plt.subplots(figsize=(16,9))
        ax.set_title('Test Dataset for Spectral Clustering', fontsize=18, fontweight='demi')
        ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
        plt.show()

    return X


if __name__ == '__main__':
    # create dataset:
    K = 2
    X = generate_dataset(visualize=False)

    # spectral clustering estimation:
    sc = SpectralClustering(n_clusters=K)
    sc.fit(X)

    category = sc.predict(X)

    # visualize:
    color = ['red','blue','green','cyan','magenta']
    labels = [f'Cluster{k:02d}' for k in range(K)]

    for k in range(K):
        plt.scatter(X[category == k][:,0], X[category == k][:,1], c=color[k], label=labels[k])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Spectral Clustering Testcase')
    plt.show()



