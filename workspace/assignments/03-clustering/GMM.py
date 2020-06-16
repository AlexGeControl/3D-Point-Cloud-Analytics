#!/opt/conda/envs/03-clustering/bin/python

# 文件功能：实现 GMM 算法

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.patches import Ellipse

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.__K = n_clusters
        self.__max_iter = max_iter

        # TODO 03: initialize GMM parameters
        self.__posteriori = None
        self.__mu = None
        self.__cov = None
        self.__priori = None


    def fit(self, data):
        """
        Estimate GMM parameters

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        Returns
        ----------
        None

        """
        N, _ = data.shape

        # init GMM params:
        self.__init_kmeans(data)

        # iterate:
        for i in range(self.__max_iter):
            # expectation:
            self.__get_expectation(data)

            # get effective count:
            effective_count = np.sum(self.__posteriori, axis=1)

            # maximization:
            self.__mu = np.asarray(
                [np.dot(self.__posteriori[k], data)/effective_count[k] for k in range(self.__K)]
            )
            self.__cov = np.asarray(
                [
                    np.dot(
                        (data - self.__mu[k]).T, np.dot(np.diag(self.__posteriori[k].ravel()), data - self.__mu[k])
                    )/effective_count[k] for k in range(self.__K)
                ]
            )
            self.__priori = (effective_count / N).reshape((self.__K, 1))


    def predict(self, data):
        """
        Classify input data

        Parameters
        ----------
        data: numpy.ndarray
            Testing set as N-by-D numpy.ndarray

        Returns
        ----------
        result: numpy.ndarray
            data labels as (N, ) numpy.ndarray

        """
        # get posteriori:
        self.__get_expectation(data)

        result = np.argmax(self.__posteriori, axis = 0)

        return result

    def get_mu(self):
        """
        Get mu
        """
        return np.copy(self.__mu)

    def __init_random(self, data):
        """
        Set initial GMM params with random initialization

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        """
        N, _ = data.shape

        # init posteriori:
        self.__posteriori = np.zeros((self.__K, N))
        # init mu:
        self.__mu = data[np.random.choice(np.arange(N), size=self.__K, replace=False)]
        # init covariances
        self.__cov = np.asarray([np.cov(data, rowvar=False)] * self.__K)
        # init priori:
        self.__priori = np.ones((self.__K, 1)) / self.__K


    def __init_kmeans(self, data):
        """
        Set initial GMM params with K-Means initialization

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        """
        N, _ = data.shape

        # init kmeans:
        k_means = KMeans(init='k-means++', n_clusters=self.__K)
        k_means.fit(data)
        category = k_means.labels_

        # init posteriori:
        self.__posteriori = np.zeros((self.__K, N))
        # init mu:
        self.__mu = k_means.cluster_centers_
        # init covariances
        self.__cov = np.asarray(
            [np.cov(data[category == k], rowvar=False) for k in range(self.__K)]
        )
        # init priori:
        value_counts = pd.Series(category).value_counts()
        self.__priori = np.asarray(
            [value_counts[k]/N for k in range(self.__K)]
        ).reshape((self.__K, 1))


    def __get_expectation(self, data):
        """
        Update posteriori

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        """
        # expectation:
        for k in range(self.__K):
            self.__posteriori[k] = multivariate_normal.pdf(
                data, 
                mean=self.__mu[k], cov=self.__cov[k]
            )
        # get posteriori:
        self.__posteriori = np.dot(
            np.diag(self.__priori.ravel()), self.__posteriori
        )
        # normalize:
        self.__posteriori /= np.sum(self.__posteriori, axis=0)

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))

    return X

if __name__ == '__main__':
    # create test set
    K = 3

    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=K)
    gmm.fit(X)
    category = gmm.predict(X)

    # visualize:
    color = ['red','blue','green','cyan','magenta']
    labels = [f'Cluster{k:02d}' for k in range(K)]

    for k in range(K):
        plt.scatter(X[category == k][:,0], X[category == k][:,1], c=color[k], label=labels[k])
    
    mu = gmm.get_mu()
    plt.scatter(mu[:,0], mu[:,1] ,s=300, c='grey', marker='P', label='Centroids')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('GMM Testcase')
    plt.show()

    

