#!/usr/bin/env python
""" learn_decoder """
import sys

import numpy as np
import sklearn.linear_model

def read_matrix(filename, sep=",", header=False, index_col=False):
    """ Read a matrix from a file, returning a numpy array.
    The file should contain one row per data point, with values separated
    by the given separator (default is comma). If header is True, the first
    line is skipped. If index_col is given, it is used as the index column
    (not included in the returned array).
    """
    lines = []
    with open(filename) as infile:
        for id, line in enumerate(infile):
            if header and (id == 0):
                continue
            if index_col:
                lines.append(list(map(float, line.strip().split(sep)[1:])))
            else:
                lines.append(list(map(float, line.strip().split(sep))))
    return np.array(lines)

#data = read_matrix("imaging_data.csv", sep=",")
#embedding_vectors = read_matrix("vectors_180concepts.GV42B300.txt", sep=" ")

def learn_decoder(data, vectors) -> np.ndarray:
     """ Given data (a CxV matrix of V voxel activations per C concepts)
     and vectors (a CxD matrix of D semantic dimensions per C concepts)
     find a matrix M such that the dot product of M and a V-dimensional 
     data vector gives a D-dimensional decoded semantic vector. 

     The matrix M is learned using ridge regression:
     https://en.wikipedia.org/wiki/Tikhonov_regularization
     """
     ridge = sklearn.linear_model.RidgeCV(
         alphas=[1, 10, .01, 100, .001, 1000, .0001, 10000, .00001, 100000, .000001, 1000000],
         fit_intercept=True,
         alpha_per_target=True,
         gcv_mode='auto'
     )
     ridge.fit(data, vectors)
     return ridge.coef_.T

def voxel_representativeness(data, vectors) -> np.ndarray:
    """ Given a CxV matrix of C concepts and V voxel activations,
    return a V-dimensional vector of voxel representativeness 
    by the following explanation from the paper:
    We learned ridge regression models (regularization parameter set to 1) to predict 
    each semantic dimension from the imaging data of each voxel and its 26 adjacent 
    neighbors in 3D, in cross-validation within the training set. 
    This yielded predicted values for each semantic dimension, which were
    then correlated with the values in the true semantic vectors. The informativeness
    score for each voxel was the maximum such correlation across dimensions.
    """
    from scipy.stats import pearsonr

    # Get the number of concepts and voxels
    C, V = data.shape

    # Initialize a vector to hold the maximum correlation for each voxel
    max_correlation = np.zeros(V)

    # Iterate over each voxel
    for v in range(V):
        # Get the voxel's data and its 26 neighbors (if available)
        voxel_data = data[:, v]
        neighbors_data = []
        
        # Collect neighboring voxels (this is a simplification, actual neighbors would depend on the 3D structure)
        for n in range(max(0, v-26), min(V, v+27)):
            if n != v:
                neighbors_data.append(data[:, n])
        
        # Combine voxel data with its neighbors
        combined_data = np.column_stack([voxel_data] + neighbors_data)

        # Iterate over each semantic dimension
        for d in range(C):
            # Compute the correlation between the combined data and the semantic dimension
            corr, _ = pearsonr(combined_data.flatten(), vectors[d, :].flatten())
            max_correlation[v] = max(max_correlation[v], abs(corr))

    return max_correlation