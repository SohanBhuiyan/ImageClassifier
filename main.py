"""
Read in the data and evaluate the models
"""

import numpy as np
from util import loadDataFile

# TODO: first, implement these functions
#       then, use them in the TODOs below


def read_faces_file(loc: str) -> np.array:
	n = 10 # number of data objects to read
	items = loadDataFile(loc, n, 60, 70)
	nparray = np.asarray(items)
	return nparray



def read_digits_file(loc: str) -> np.array:
	n = 10 # number of data objects to read
	items = loadDataFile(loc, n, 28, 28)
	nparray = np.asarray(items)
	return nparray



def faces_features(faces: np.array) -> np.array:
	raise NotImplementedError


def digits_features(digits: np.array) -> np.array:
	raise NotImplementedError


# TODO: read the raw input data and label data for the faces
# NOTE: keep the train, test, and validation data separate

# TODO: turn the raw input data into a 2d numpy array where each row describes an input face and each column is a feature

# TODO: turn the label data into a numpy vector

# TODO: train a perceptron on the training data
# TODO: evaluate the perceptron on the validation data

# TODO: train a naive bayes model on the training data
# TODO: evaluate the naive bayes model on the validation data


# TODO: redo this for digits by making a multi-class NB and Perceptron model
