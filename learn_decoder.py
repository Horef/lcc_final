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


def learn_encoder(voxels, vectors) -> np.ndarray:
    """ Given voxels (a CxV matrix of V voxel activations per C concepts)
    and vectors (a CxD matrix of D semantic dimensions per C concepts)
    find a matrix M such that the dot product of a D-dimensional semantic vector
    and M gives a V-dimensional encoded voxel activations.
    
    The matrix M is learned using ridge regression.
    """
    ridge = sklearn.linear_model.RidgeCV(
        alphas=[1, 10, .01, 100, .001, 1000, .0001, 10000, .00001, 100000, .000001, 1000000],
        fit_intercept=True,
        alpha_per_target=True,
        gcv_mode='auto'
    )
    ridge.fit(vectors, voxels)
    return ridge.coef_.T