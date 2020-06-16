#!/opt/conda/envs/03-clustering/bin/python

# 文件功能： 实现 K-Means 算法

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class KMeans(object):
    """
    KMeans with both random and KMeans++ initialization

    Parameters
    ----------
    n_clusters: int
        Number of clusters 
    tolerance: float
        Initial splitting axis
    max_iter: int
        Maximum number of iterations

    Attributes
    ----------

    """
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.01, max_iter=300):
        self.__K = n_clusters
        self.__tolerance = tolerance
        self.__max_iter = max_iter
        self.__centroids = None

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
        # TODO 01: implement KMeans fit 
        # get input size:
        N, D = data.shape
        # format as pandas dataframe:
        __data = pd.DataFrame(
            data = data,
            index = np.arange(N),
            columns = [f'x{i:03d}' for i in range(D)]
        )
        __data['cluster'] = 0

        # get tolerance:
        self.__tolerance = KMeans.__tolerance(data, self.__tolerance)
        # get initial centroids:
        self.__centroids = self.__get_init_centroid_kmeanspp(data)

        # iterate:
        for i in range(self.__max_iter):
            # expectation:
            __data.cluster = __data.apply(
                lambda x: KMeans.__assign(x[:-1].values, self.__centroids),
                axis = 1
            )
            # maximization:
            new_centroids = __data.groupby(['cluster']).mean().values
            
            # evaluate squared diff:
            diff = (new_centroids - self.__centroids).ravel()
            squared_diff = np.dot(diff, diff)
            

            # update centroids:
            self.__centroids = new_centroids

            # early stopping check:
            if squared_diff <= self.__tolerance:
                print(f'[KMeans - Fit]: early stopping with squared centroids diff {squared_diff:.2f} at iteration {i:03d}')
                break

    def predict(self, data):
        """
        Classify input data

        Parameters
        ----------
        data: numpy.ndarray
            Testing set as N-by-D numpy.ndarray

        Returns
        ----------
        matches: numpy.ndarray
            potential matches as N-by-2 numpy.ndarray

        """
        # TODO 02: implement KMeans predict
        N, _ = data.shape

        result = np.asarray(
            [KMeans.__assign(data[i], self.__centroids) for i in range(N)]
        )

        return result

    def get_centroids(self):
        """
        Get centroids

        Parameters
        ----------
        None

        Returns
        ----------
        centroids: numpy.ndarray
            cluster centroids as numpy.ndarray

        """
        return np.copy(self.__centroids)

    def __get_init_centroid_random(self, data):
        """
        Get initial centroids using random selection

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        """
        N, _ = data.shape

        idx_centroids = np.random.choice(np.arange(N), size=self.__K, replace=False)

        centroids = data[idx_centroids]

        return centroids

    def __get_init_centroid_kmeanspp(self, data):
        """
        Get initial centroids using KMeans++ selection

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        """
        N, _ = data.shape

        # select the first centroid by random choice:
        centroids = data[np.random.choice(np.arange(N), size=1, replace=False)]

        # for the remaining centroids, select by prob based on minimum distance to existing centroids:
        for _ in range(1, self.__K):
            # find minimum distance to existing centroids for each poit
            distances = np.asarray(
                [
                    np.min(np.linalg.norm(d - centroids, axis=1))**2 for d in data
                ]
            )
            # generate cumulative probability:
            probs = distances / np.sum(distances)
            cum_probs = np.cumsum(probs)
            # select new centroid:
            centroids = np.vstack(
                (centroids, data[np.searchsorted(cum_probs, random.random())])
            )
        
        return centroids

    @staticmethod
    def __assign(data, centroids):
        """
        Assign data point to centroids of minimum L2 distance

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray
        centroids: numpy.ndarray
            Centroids as N-by-D numpy.ndarray

        """
        return np.argmin(np.linalg.norm(centroids - data, axis=1))

    @staticmethod
    def __tolerance(data, tol):
        """
        Return a tolerance which is independent of the dataset
        """
        variances = np.var(data, axis=0)
        return np.mean(variances) * tol

if __name__ == '__main__':
    # create test set:
    K = 2

    X = np.array(
        [
            [1, 2], 
            [1.5, 1.8], 
            [5, 8], 
            [8, 8], 
            [1, 0.6], 
            [9, 11]
        ]
    )

    # fit:
    k_means = KMeans(n_clusters=K)
    k_means.fit(X)
    # predict:
    category = k_means.predict(X)

    # visualize:
    color = ['red','blue','green','cyan','magenta']
    labels = [f'Cluster{k:02d}' for k in range(K)]
    for k in range(K):
        plt.scatter(X[category == k][:,0], X[category == k][:,1], c=color[k], label=labels[k])
    
    centroids = k_means.get_centroids()
    plt.scatter(centroids[:,0], centroids[:,1] ,s=300, c='grey', marker='P', label='Centroids')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('KMeans Testcase')
    plt.show()

