import numpy as np
import torch

def build_adjacency_matrix(zone_pairs_df, num_zones=263):
    """
    zone_pairs_df contains columns: ['PULocationID', 'DOLocationID', 'trip_count']
    """
    A = np.zeros((num_zones, num_zones), dtype=np.float32)
    
    for _, row in zone_pairs_df.iterrows():
        u = int(row['PULocationID']) - 1  # 0-indexed
        v = int(row['DOLocationID']) - 1
        if 0 <= u < num_zones and 0 <= v < num_zones:
            A[u, v] += row['trip_count']
            A[v, u] += row['trip_count']  # Symmetric graph

    # Apply degree normalization: D^(-1/2) * A * D^(-1/2)
    row_sum = A.sum(axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
    d_inv_sqrt[row_sum == 0] = 0.0
    D_mat = np.diag(d_inv_sqrt)
    
    A_norm = D_mat.dot(A).dot(D_mat)
    return torch.tensor(A_norm, dtype=torch.float32)