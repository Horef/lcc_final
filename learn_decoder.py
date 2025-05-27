#!/usr/bin/env python
""" learn_decoder """
import sys

import numpy as np
import sklearn.linear_model

def read_matrix(filename, sep=","):
    lines = []
    with open(filename) as infile:
        for line in infile:
            lines.append(list(map(float, line.strip().split(sep))))
    return np.array(lines)

data = read_matrix("imaging_data.csv", sep=",")
embedding_vectors = read_matrix("vectors_180concepts.GV42B300.txt", sep=" ")

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
         fit_intercept=False
     )
     ridge.fit(data, vectors)
     return ridge.coef_.T

