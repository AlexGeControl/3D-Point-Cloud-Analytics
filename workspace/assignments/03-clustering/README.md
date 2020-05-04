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

I have followed the practice of OOAD and refactored the original implementation. The code is available at **/workspace/assignments/02-nearest-neighbor/octree.py**

First is the code for **tree building**. The attribute **depth** has been added to OCTree.Octant.

```python
    def __build(self):
        """
        Build KDTree
        """
        return self.__do_build(
            depth = 0,
            point_indices = np.arange(self.__N)
        )

    def __do_build(self, depth, point_indices):
        """
        Node for KDTree

        Parameters
        ----------
        depth : int
            Current depth
        point_indices: list of int
            Points indices associated with this node

        """
        # create node:
        mins = np.amin(self.__point_cloud[point_indices], axis=0)
        maxs = np.amax(self.__point_cloud[point_indices], axis=0)
        center = (mins + maxs) / 2
        extent = np.max(maxs - mins) / 2
        
        # create new node: 
        root = OCTree.Octant(
            depth = depth + 1,
            center = center, 
            extent = extent, 
            point_indices = point_indices,
            children = [None] * 8
        )

        # update height:
        if (root.depth > self.__height):
            self.__height = root.depth

        # split if the node is oversize:
        N = len(point_indices)
        if N > self.__leaf_size and extent > self.__min_extent:
            root.is_leaf = False

            # generate morton codes:
            morton_codes = [
                (point[0] > root.center[0]) * 4 + (point[1] > root.center[1]) * 2 + (point[2] > root.center[2]) * 1
                for point in self.__point_cloud[point_indices]
            ]

            # assign points to children:
            children_point_indices = {}
            for i, idx in zip(morton_codes, point_indices):
                if children_point_indices.get(i, None) is None:
                    children_point_indices[i] = [idx]
                else:
                    children_point_indices[i].append(idx)
            
            for i in children_point_indices:
                root.children[i] = self.__do_build(
                    depth = root.depth, 
                    point_indices = children_point_indices[i]
                )

        return root
```

Next comes the key logic for nearest neighbor search. The implementations of kNN and RNN are just wrappers around this logic with different **result_set** implementation.

```python
    def __do_search(self, root, query, result_set, fast_rnn=False):
        """ 
        kNN search implementation

        Parameters
        ----------
        root: KDTree.Node
            KDTree root
        query: numpy.ndarray
            query point
        result_set: KNNResultSet or RadiusNNResultSet
            result set

        """
        if root is None:
            return False

        # contain check for root:
        if fast_rnn and OCTree.__contains(query, result_set.worstDist(), root):
            for index in root.point_indices:
                result_set.add_point(result_set.worstDist(), index)             
            return True

        if root.is_leaf and len(root.point_indices) > 0:
            dists = np.linalg.norm(
                self.__point_cloud[root.point_indices] - query, 
                axis = 1
            )
            for dist, index in zip(dists, root.point_indices):
                result_set.add_point(dist, index) 
        else:
            # identify octant:
            morton_code = (query[0] > root.center[0]) * 4 + (query[1] > root.center[1]) * 2 + (query[2] > root.center[2]) * 1
            
            if self.__do_search(root.children[morton_code], query, result_set):
                return True
            
            for i, child in enumerate(root.children):
                if i == morton_code or child is None:
                    continue
                
                if not OCTree.__overlaps(query, result_set.worstDist(), child):
                    continue

                if self.__do_search(child, query, result_set):
                    return True

        return OCTree.__inside(query, result_set.worstDist(), root)
```

### 4. Testing and Benchmarking

#### Implementation Logic Check

To ensure the correctness of KDTree & OCTree implementation, the following tests are used. For each kNN or RNN implementation, its output will be compared with that of brute-force for 100 random queries. One sample test case is as follows:

```python
    # random query test:
    for _ in range(100):
        # generate query point:
        query = np.random.rand(D)

        # 01 -- knn: brute-force as baseline:
        dists = np.linalg.norm(point_cloud - query, axis=1)
        sorting_idx = np.argsort(dists)
        brute_force_result = {i for i in sorting_idx[:k]}

        knn_result_set = KNNResultSet(capacity=k)
        kd_tree.knn_search(query, knn_result_set)
        knn_result = {i.index for i in knn_result_set.dist_index_list}
        assert len(brute_force_result - knn_result) == 0

        # 02 -- rnn: brute-force as baseline:
        dists = np.linalg.norm(point_cloud - query, axis=1)
        brute_force_result = {i for i, d in enumerate(dists) if d <= r}

        rnn_result_set = RadiusNNResultSet(radius = r)
        kd_tree.rnn_search(query, rnn_result_set)
        rnn_result = {i.index for i in rnn_result_set.dist_index_list}
        assert len(brute_force_result - rnn_result) == 0

    print('[KDTree kNN & RNN Random Query Test]: Successful')
```

To run the test cases, go to **/workspace/assignments/02-nearest-neighbor/** and run the following commands:

```bash
# activate environment:
source activate point-cloud
# go to HW2 working dir:
cd assignments/02-nearest-neighbor/
# kdtree:
python kdtree.py
# octree:
python octree.py
```

The results are as follows, which indicates the implementation is correct

<img src="doc/01-kdtree-octreee-unit-test.png" alt="Unit Tests">

#### Benchmark

To run the test cases, go to **/workspace/assignments/02-nearest-neighbor/** and run the following commands:

```bash
# perform benchmark analysis on KITTI:
python benchmark.py -i /workspace/data/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/
```

The results are as follows, which indicates the implementation is correct

<img src="doc/02-benchmark.png" alt="Benchmark">

Using Python KDTree is slightly faster than OCTree. I also noticed that sometimes the use of contains could lower the performance. I will try to further debug during the remaining time.