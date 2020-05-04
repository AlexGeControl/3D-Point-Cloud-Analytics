# Clustering

Python implementation of KMeans++, GMM and Spectral Clustering for clustering algorithm analysis.

---

## Homework Solution

### 1. Get Clustering Dataset 

TBD

### 2. KMeans

The implementation, which follows the practice of Python OOAD, is available at (click to follow the link) **[/workspace/assignments/03-clustering/KMeans.py](KMeans.py)**

First is the code for **fit**. Here the training set is further formatted as Pandas dataframe for easy analytics inspired by corresponding map-reduce implementation.

```python
    def fit(self, data):
        """
        Estimate the K centroids

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

        """
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
        self.__centroids = self.__get_init_centroid_random(data)

        # iterate:
        for iter in range(self.__max_iter):
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
                print(f'[KMeans - Fit]: early stopping with squared centroids diff {squared_diff:.2f} at iteration {iter:03d}')
                break
```

Next comes the logic for **predict**. Here brute-force method based on numpy intrinsic parallel computing is used for simplicity.

```python
    def predict(self, data):
        """
        Classify input data

        Parameters
        ----------
        data: numpy.ndarray
            Testing set as N-by-D numpy.ndarray

        """
        N, _ = data.shape

        result = np.asarray(
            [KMeans.__assign(data[i], self.__centroids) for i in range(N)]
        )

        return result    
```

To run the test case, launch the docker environment and go to **/workspace/assignments/03-clustering/** and run the following commands:

```bash
# go to HW3 working dir:
cd /workspace/assignments/03-clustering
# activate environment:
source activate point-cloud
# KMeans:
./KMeans.py
```

The result clustering is as follows:

<img src="doc/01-kmeans-testcase.png" alt="KMeans Testcase" width="100%">

### 3. GMM

The implementation, which follows the practice of Python OOAD, is available at (click to follow the link) **[/workspace/assignments/03-clustering/GMM.py](GMM.py)**

First is the code for **fit**. Here the GMM params are initialized using KMeans++. GMM is in essence a local optimization algorithm thus a good initialization does matter.

```python
    def fit(self, data):
        """
        Estimate GMM parameters

        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray

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
        ).reshape((3, 1))
```

Next comes the logic for **predict**. Here category is estimated using Maximum-A-Posteriori(MAP).

```python
    def predict(self, data):
        """
        Classify input data

        Parameters
        ----------
        data: numpy.ndarray
            Testing set as N-by-D numpy.ndarray

        """
        # get posteriori:
        self.__get_expectation(data)
        
        result = np.argmax(self.__posteriori, axis = 0)

        return result 
```

To run the test case, launch the docker environment and go to **/workspace/assignments/03-clustering/** and run the following commands:

```bash
# go to HW3 working dir:
cd /workspace/assignments/03-clustering
# activate environment:
source activate point-cloud
# GMM:
./GMM.py
```

The result clustering is as follows:

<img src="doc/02-gmm-testcase.png" alt="GMM Testcase" width="100%">

---

#### Benchmark

To run the test cases, go to **/workspace/assignments/02-nearest-neighbor/** and run the following commands:

```bash
# perform benchmark analysis on KITTI:
python benchmark.py -i /workspace/data/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/
```

The results are as follows, which indicates the implementation is correct

<img src="doc/02-benchmark.png" alt="Benchmark">

Using Python KDTree is slightly faster than OCTree. I also noticed that sometimes the use of contains could lower the performance. I will try to further debug during the remaining time.