# 文件功能： 实现 K-Means 算法

import numpy as np

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始

        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始

        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

