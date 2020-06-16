#!/opt/conda/envs/02-nearest-neighbor/bin/python

# kdtree implementation
from collections import namedtuple
import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet

class KDTree:
    """
    KDTree for kNN & RNN search

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Point cloud measurements as N-by-3 numpy.ndarray
    axis: int
        Initial splitting axis
    leaf_size: int
        The maximum size of leaf node

    Attributes
    ----------

    """
    class Node:
        """
        Node in KDTree

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
        def __init__(self, axis, depth, value = None, left = None, right = None, point_indices = None):
            self.axis = axis
            self.depth = depth
            self.value = value
            self.left = left
            self.right = right
            self.point_indices = point_indices

        def is_leaf(self):
            """
            Whether the node is a leaf

            """
            return True if self.value is None else False

        def __str__(self):
            output = ''
            output += 'depth %d, ' % self.depth
            output += 'axis %d, ' % self.axis
            if self.value is None:
                output += 'split value: leaf, '
            else:
                output += 'split value: %.2f, ' % self.value
            output += 'point_indices: '
            output += '[]' if self.point_indices is None else str(self.point_indices.tolist())
            return output


    def __init__(self, point_cloud, init_axis=0, leaf_size=4):
        # parse point cloud:
        self.__point_cloud = point_cloud
        self.__N, self.__D = point_cloud.shape
        # parse tree config:
        self.__init_axis = init_axis
        self.__leaf_size = leaf_size
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
        self.__do_search(self.__root, query, knn_result_set)

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
        self.__do_search(self.__root, query, rnn_result_set)

    def __build(self):
        """
        Build KDTree
        """
        return self.__do_build(
            axis = self.__init_axis,
            depth = 0,
            point_indices = np.arange(self.__N)
        )

    def __do_build(self, axis, depth, point_indices):
        """
        Node for KDTree

        Parameters
        ----------
        axis : int
            Splitting axis
        depth : int
            Current depth
        point_indices: list of int
            Points indices associated with this node
        """
        # create new node: 
        root = KDTree.Node(
            axis = axis, 
            depth = depth + 1,
            point_indices = point_indices
        )

        # update height:
        if (root.depth > self.__height):
            self.__height = root.depth

        # split if the node is oversize:
        N = len(point_indices)
        if N > self.__leaf_size:
            # get median:
            sorting_indices, sorted_values = KDTree.__sort_key_by_value(
                key = point_indices,
                value = self.__point_cloud[point_indices, axis]
            )
            median = (sorted_values[N // 2 - 1] + sorted_values[N // 2]) / 2

            # split:
            root.value = median
            next_axis = self.__get_next_axis(axis)
            root.left = self.__do_build(
                axis = next_axis,
                depth = root.depth,
                point_indices = sorting_indices[:N // 2]
            )
            root.right = self.__do_build(
                axis = next_axis,
                depth = root.depth,
                point_indices = sorting_indices[N // 2:]
            )
        
        return root

    @staticmethod
    def __sort_key_by_value(key, value):
        """
        Sort key by value

        Parameters
        ----------
        key: numpy.ndarray
            value indices
        value: numpy.ndarray
            values

        """
        # key must be 1-d numpy.ndarray:
        assert len(key.shape) == 1
        # key & value must have the same shape:
        assert key.shape == value.shape
        
        sorting_idx = np.argsort(value)
        key_sorted = key[sorting_idx]
        value_sorted = value[sorting_idx]

        return key_sorted, value_sorted

    def __get_next_axis(self, axis):
        """ 
        Generate next axis for KDTree splitting using round-robin

        Parameters
        ----------
        axis: int
            splitting index

        """
        return (axis + 1) % self.__D

    def __do_traverse(self, root):        
        # display current node:
        print(root)

        # dive in for non-leaf node:
        if not root.is_leaf():
            self.__do_traverse(root.left)
            self.__do_traverse(root.right)

    def __do_search(self, root, query, result_set):
        """ 
        kNN & RNN search implementation

        Parameters
        ----------
        root: KDTree.Node
            KDTree root
        query: numpy.ndarray
            query point
        result_set: KNNResultSet or RadiusNNResultSet
            result set

        """        
        if root.is_leaf():
            dists = np.linalg.norm(
                self.__point_cloud[root.point_indices] - query, 
                axis = 1
            )
            for dist, index in zip(dists, root.point_indices):
                result_set.add_point(dist, index)
        else:
            if query[root.axis] <= root.value:
                self.__do_search(root.left, query, result_set)
                if (result_set.worstDist() > np.fabs(query[root.axis] - root.value)):
                    self.__do_search(root.right, query, result_set)
            else:
                self.__do_search(root.right, query, result_set)
                if (result_set.worstDist() > np.fabs(query[root.axis] - root.value)):
                    self.__do_search(root.left, query, result_set)                


def main():
    # configuration
    N = 64000
    D = 3
    leaf_size = 4
    k = 8
    r = 0.372

    point_cloud = np.random.rand(N, D)

    kd_tree = KDTree(
        point_cloud = point_cloud, 
        init_axis = 0, 
        leaf_size = leaf_size
    )
    # kd_tree.traverse()

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


if __name__ == '__main__':
    main()