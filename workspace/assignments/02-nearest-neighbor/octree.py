#!/opt/conda/envs/02-nearest-neighbor/bin/python

# octree implementation
import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet

class OCTree:
    """
    OCTree for kNN & RNN search

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Point cloud measurements as N-by-3 numpy.ndarray
    leaf_size: int
        The maximum size of leaf node

    Attributes
    ----------

    """
    class Octant:
        """
        Octant in OCTree

        Parameters
        ----------
        axis : int
            Splitting axis
        value: int
            Splitting value
        left: KDTree.Node
            Left child node
        right: KDTree.Node
            Right child node
        point_indices: list of int
            Point indices associated with this node
        """
        def __init__(self, depth, center, extent, point_indices, is_leaf = True, children = [None]*8):
            self.depth = depth
            self.center = center
            self.extent = extent
            self.point_indices = point_indices
            self.is_leaf = is_leaf
            self.children = children

        def __str__(self):
            output = ''
            output += 'depth: %d, ' % self.depth
            output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
            output += 'extent: %.2f, ' % self.extent
            output += 'point_indices: ' + str(self.point_indices)
            output += 'is_leaf: %d, ' % self.is_leaf
            output += 'children: ' + str([x is not None for x in self.children])
            return output

    def __init__(self, point_cloud, leaf_size=4, min_extent=0.1):
        # parse point cloud:
        self.__point_cloud = point_cloud
        self.__N, self.__D = point_cloud.shape
        # parse tree config:
        self.__leaf_size = leaf_size
        self.__min_extent = min_extent
        # init tree:
        self.__height = 0
        self.__root = self.__build()

    def traverse(self):
        """
        Traverse KDTree
        """
        self.__do_traverse(self.__root)

    def knn_search(self, query, knn_result_set):
        """
        K nearest neighbor search

        Parameters
        ----------
        query : numpy.ndarray
            query point
        knn_result_set : KNNResultSet
            kNN result set

        """
        self.__do_search(self.__root, query, knn_result_set, fast_rnn=False)

    def rnn_search(self, query, rnn_result_set):
        """
        Radius based nearest neighbor search

        Parameters
        ----------
        query : numpy.ndarray
            query point
        rnn_result_set : RadiusNNResultSet
            RNN result set

        """        
        self.__do_search(self.__root, query, rnn_result_set, fast_rnn=False)

    def rnn_fast_search(self, query, rnn_result_set):
        """
        Radius based nearest neighbor search in a faster way

        Parameters
        ----------
        query : numpy.ndarray
            query point
        rnn_result_set : RadiusNNResultSet
            RNN result set

        """    
        self.__do_search(self.__root, query, rnn_result_set, fast_rnn=True)

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

    def __do_traverse(self, root):        
        # display current node:
        print(root)

        # dive in for non-None children:
        for child in root.children:
            if child is not None:
                self.__do_traverse(child)

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

    @staticmethod
    def __inside(query, radius, octant):
        """
        Whether the query is inside given octant

        Parameters
        ----------
        query: numpy.ndarray
            query point
        radius: float
            query radius
        octant: OCTree.Octant
            octant selected 
        """
        dists = np.fabs(query - octant.center)
        return np.all(octant.extent - dists > radius)

    @staticmethod
    def __overlaps(query, radius, octant):
        """
        Whether the query overlaps with given octant

        Parameters
        ----------
        query: numpy.ndarray
            query point
        radius: float
            query radius
        octant: OCTree.Octant
            octant selected 
        """
        dists = np.fabs(query - octant.center)

        # completely outside, since query is outside the relevant area
        if np.max(dists) > (radius + octant.extent):
            return False

        # if pass the above check, consider the case that the ball is contacting the face of the octant
        if np.sum((dists < octant.extent).astype(np.int)) >= 2:
            return True

        # conside the case that the ball is contacting the edge or corner of the octant
        # since the case of the ball center (query) inside octant has been considered,
        # we only consider the ball center (query) outside octant
        x_diff = max(dists[0] - octant.extent, 0)
        y_diff = max(dists[1] - octant.extent, 0)
        z_diff = max(dists[2] - octant.extent, 0)

        return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius

    @staticmethod
    def __contains(query, radius, octant):
        """
        Whether the query contains the given octant

        Parameters
        ----------
        query: numpy.ndarray
            query point
        radius: float
            query radius
        octant: OCTree.Octant
            octant selected 
        """
        dists = np.fabs(query - octant.center)

        return np.linalg.norm(dists + octant.extent) < radius


def main():
    # configuration
    N = 64000
    D = 3
    leaf_size = 4
    min_extent = 0.05
    k = 8
    r = 0.372

    # generate point cloud:
    point_cloud = np.random.rand(N, D)

    octree = OCTree(
        point_cloud=point_cloud, 
        leaf_size=leaf_size, 
        min_extent=min_extent
    )
    # octree.traverse()

    # random query test:
    for _ in range(100):
        # generate query point:
        query = np.random.rand(D)

        # 01 -- knn: brute-force as baseline:
        dists = np.linalg.norm(point_cloud - query, axis=1)
        sorting_idx = np.argsort(dists)
        brute_force_result = {i for i in sorting_idx[:k]}

        knn_result_set = KNNResultSet(capacity=k)
        octree.knn_search(query, knn_result_set)
        knn_result = {i.index for i in knn_result_set.dist_index_list}
        assert len(brute_force_result - knn_result) == 0

        # 02 -- rnn: brute-force as baseline:
        dists = np.linalg.norm(point_cloud - query, axis=1)
        brute_force_result = {i for i, d in enumerate(dists) if d <= r}

        rnn_result_set = RadiusNNResultSet(radius = r)
        octree.rnn_search(query, rnn_result_set)
        rnn_result = {i.index for i in rnn_result_set.dist_index_list}
        assert len(brute_force_result - rnn_result) == 0

    print('[OCTree kNN & RNN Random Query Test]: Successful')

    begin_t = time.time()
    print("[OCTree]: RNN search normal:")
    for i in range(100):
        query = np.random.rand(3)
        rnn_result_set = RadiusNNResultSet(radius=0.5)
        octree.rnn_search(query, rnn_result_set)
    # print(result_set)
    print("\tSearch takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    begin_t = time.time()
    print("[OCTree]: RNN search fast:")
    for i in range(100):
        query = np.random.rand(3)
        rnn_result_set = RadiusNNResultSet(radius = 0.5)
        octree.rnn_fast_search(query, rnn_result_set)
    # print(result_set)
    print("\tSearch takes %.3fms\n" % ((time.time() - begin_t)*1000))


if __name__ == '__main__':
    main()