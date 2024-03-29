from typing import List

import numpy as np
import torch
from sklearn.metrics import pairwise_distances_argmin_min


from itertools import cycle, islice


def roundrobin(*iterables):
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

class Clustering_Processor:
    """
    A processor class that makes it easy to obtain indices from clusters with
    various methods
    """

    labels: np.array
    data_pct: float
    num_clusters: int
    cluster_num: int

    def __init__(self, cluster):
        self.labels = cluster["labels_"]
        self.kmeans_cluster_centers = cluster["cluster_centers_"]

    def get_cluster_indices(self, cluster_num: int):
        return np.where(self.labels == cluster_num)[0]

    def get_diverse_stream(self):
        total_indices = []
        for i in range(len(self.kmeans_cluster_centers)):
            total_indices.append(self.get_cluster_indices(i))
        return list(roundrobin(*total_indices))

    def get_cluster_indices_by_pct(self, data_pct: float, original_len: int) -> List:
        """
        Input:
            data_pct: specify how many elements are required from clusters
            original_len: length of the dataset
        Output:
            cluster_indices: cluster indices

        This method return concatenated cluster indices whose propotion equals len(dataset)*data_percentage
        """
        current_len, cluster_indices = 0, []
        for i in set(self.labels):
            curr_cluster_indices = self.get_cluster_indices(i)
            current_len += len(curr_cluster_indices)
            if current_len <= int(original_len * data_pct):
                cluster_indices.extend(curr_cluster_indices)
            else:
                return cluster_indices

    def get_cluster_indices_by_num(self, num_clusters: int) -> List:
        """
        Input:
            num_clusters: specify how many clusters to return
        Output:
            cluster_indices: cluster indices

        This method returns concatenated cluster indices whose propotion equals to that of number of elements in specified number of cluster
        """
        indices = []
        for i in range(num_clusters):
            indices.extend(self.get_cluster_indices(i))
        return indices

    def get_cluster_indices_from_centroid(self, embeddings: torch.tensor) -> np.array:
        return pairwise_distances_argmin_min(self.kmeans_cluster_centers, embeddings)[0]
