from __future__ import annotations
from sklearn.decomposition import PCA
import numpy as np
import impy as ip

def fit_pca(
    input: ip.ImgArray,
    binary_mask: ip.ImgArray | None = None,
    n_components: int = 3
) -> PCA:
    pca = PCA(n_components=n_components)
    if binary_mask is not None:
        if input.shape[1:] != binary_mask.shape:
            raise TypeError("Shape mismatch between input images and mask.")
        X = np.stack([ar[binary_mask] for ar in input.value], axis=0)
    else:
        X = input.reshape(3, -1).T
        
    converted_data = pca.fit_transform(X)
    
    # plt.scatter(converted_data[:, 0], converted_data[:, 1])
    return pca
