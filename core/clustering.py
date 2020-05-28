from typing import Dict, List, Optional

import numpy as np
import sklearn


class Clustering_Processor:
    """
    A processor class that makes it easy to obtain indices from clusters with
    various methods
    """

    labels: np.array
    data_pct: float
    num_clusters: int
    cluster_num: int

    def __init__(
        self, cluster: sklearn.cluster, data_pct: Optional[float] = None, num_clusters: Optional[int] = None,
    ):
        self.labels = cluster["labels_"]
        self.data_pct = data_pct
        self.num_clusters = num_clusters

    def get_cluster_indices(self, cluster_num: int):
        return np.where(self.labels == cluster_num)[0]

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
            if current_len < int(original_len * data_pct):
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
