"""
DBSCAN-based MSA clustering for diverse sequence sampling in ProteinTTT.

Adapted from AF_Cluster:
https://github.com/HWaymentSteele/AF_Cluster/blob/main/scripts/ClusterMSA.py

Reference:
    Wayment-Steele, H. K., et al. (2024).
    Predicting multiple conformations via sequence clustering and AlphaFold2.
    Nature, 625(7996), 832-839.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def _one_hot_encode(tokens: np.ndarray, num_classes: int = None) -> np.ndarray:
    """One-hot encode a token matrix into flat feature vectors for DBSCAN.

    Args:
        tokens: Integer token array [num_sequences, sequence_length].
        num_classes: Alphabet size. Inferred from data if None.

    Returns:
        One-hot matrix [num_sequences, sequence_length * num_classes].
    """
    n_seqs, seq_len = tokens.shape
    if num_classes is None:
        num_classes = int(tokens.max()) + 1
    ohe = np.eye(num_classes, dtype=np.float32)[tokens]  # [n_seqs, seq_len, num_classes]
    return ohe.reshape(n_seqs, -1)


def cluster_msa(
    tokens: np.ndarray,
    eps: float = None,
    min_samples: int = 3,
    min_eps: float = 3.0,
    max_eps: float = 20.0,
    eps_step: float = 0.5,
) -> np.ndarray:
    """Cluster tokenized MSA sequences using DBSCAN.

    When *eps* is None the optimal value is selected by scanning a range on a
    25 % subset of the data (following the AF_Cluster heuristic).

    Args:
        tokens: Tokenized MSA [num_sequences, sequence_length] (numpy array).
        eps: Fixed DBSCAN epsilon.  If None, scans ``[min_eps, max_eps]``.
        min_samples: Minimum number of samples for a DBSCAN core point.
        min_eps: Start of epsilon scan range.
        max_eps: End of epsilon scan range.
        eps_step: Step size for the epsilon scan.

    Returns:
        Integer array [num_sequences] with cluster IDs (>= 0) or -1 for
        unclustered sequences.
    """
    ohe = _one_hot_encode(tokens)

    if eps is None:
        # Scan eps on a random 25 % subset to pick the value that maximises
        # the number of clusters (same heuristic as AF_Cluster).
        eps_vals = np.arange(min_eps, max_eps + eps_step, eps_step)
        n_clusters_list: list[int] = []
        subset_size = max(min_samples, len(ohe) // 4)
        subset_idx = np.random.choice(len(ohe), size=min(subset_size, len(ohe)), replace=False)
        subset = ohe[subset_idx]

        for e in eps_vals:
            labels = DBSCAN(eps=e, min_samples=min_samples).fit_predict(subset)
            n_clust = len(set(labels) - {-1})
            n_clusters_list.append(n_clust)
            if e > 10 and n_clust <= 1:
                break

        if not n_clusters_list or max(n_clusters_list) == 0:
            return np.full(len(tokens), -1, dtype=int)

        eps = float(eps_vals[np.argmax(n_clusters_list)])

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(ohe)
    return labels
