# 该文件定义了在树中查找数据所需要的数据结构，类似一个中间件

import copy


class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):
        return self.distance < other.distance


class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        self.worst_dist = 1e10
        self.dist_index_list = []
        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist, 0))

        self.comparison_counter = 0

    def size(self):
        return self.count

    def full(self):
        return self.count == self.capacity

    def worstDist(self):
        return self.worst_dist

    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.worst_dist:
            return

        if self.count < self.capacity:
            self.count += 1

        i = self.count - 1
        while i > 0:
            if self.dist_index_list[i-1].distance > dist:
                self.dist_index_list[i] = copy.deepcopy(self.dist_index_list[i-1])
                i -= 1
            else:
                break

        self.dist_index_list[i].distance = dist
        self.dist_index_list[i].index = index
        self.worst_dist = self.dist_index_list[self.capacity-1].distance
        
    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d comparison operations.' % self.comparison_counter
        return output


class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.count = 0
        self.worst_dist = radius
        self.dist_index_list = []

        self.comparison_counter = 0

    def size(self):
        return self.count

    def worstDist(self):
        return self.radius

    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.radius:
            return

        self.count += 1
        self.dist_index_list.append(DistIndex(dist, index))

    def __str__(self):
        self.dist_index_list.sort()
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d neighbors within %f.\nThere are %d comparison operations.' \
                  % (self.count, self.radius, self.comparison_counter)
        return output


