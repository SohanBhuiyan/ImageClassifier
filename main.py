"""
Read in the data and evaluate the models
"""

import numpy as np
import util
from nb import NaiveBayes
from perceptron import Perceptron
from knn import KNN
from features import rawPixelFeature, rowPixelFeature, gridFeature, multipleGridFeatures


# TODO: first, implement these functions
#       then, use them in the TODOs below


def read_faces_file(loc: str, n: int) -> np.array:
	items = util.loadDataFile(loc, n, 60, 70)
	nparray = np.asarray(items)
	return nparray



def read_digits_file(loc: str, n: int) -> np.array:
	items = util.loadDataFile(loc, n, 28, 28)
	nparray = np.asarray(items)
	return nparray


def perceptron_faces_features(faces: np.array) -> np.array:
	features = []

	for face in faces:
		feature_vector = rawPixelFeature(face.getPixels())
		features.append(feature_vector)
	# convert to nparray
	features = np.array(features)
	return features


def knn_faces_features(faces: np.array) -> np.array:
	features = []

	for face in faces:
		arr = face.getPixels()
		feature_vector = [arr.mean(), arr.std(), len(arr[arr != 0])]
		feature_vector = gridFeature(face.getPixels())
		features.append(feature_vector)
	# convert to nparray
	features = np.array(features)
	return features

def nb_faces_features(faces: np.array) -> np.array:
	features = []

	for face in faces:
		feature_vector = multipleGridFeatures(face.getPixels())
		features.append(feature_vector)

	features = np.array(features)
	return features


def digits_features(digits: np.array) -> np.array:
	raise NotImplementedError

def split(array):
	s = np.split(array,2)
	print(s)
	return s

if __name__ == '__main__':
	n_samples = 450
	n_validation = n_samples // 10
	faces_list = read_faces_file("facedata/facedatatrain", n_samples)

	y = np.array(util.loadLabelsFile("facedata/facedatatrainlabels", n_samples))
	train_y = y[:-n_validation]
	validation_y = y[-n_validation:]

	x = perceptron_faces_features(faces_list)
	train_x = x[:-n_validation]
	validation_x = x[-n_validation:]
	p_model = Perceptron(100)
	p_model.train(train_x, train_y)
	matches = list(p_model.predict(validation_x) == validation_y).count(True)
	accuracy = matches / n_validation
	print('Perceptron accuracy:', 100 * accuracy, '%')

	x = nb_faces_features(faces_list)
	train_x = x[:-n_validation]
	validation_x = x[-n_validation:]
	nb_model = NaiveBayes()
	nb_model.train(train_x, train_y)
	matches = list(nb_model.predict(validation_x) == validation_y).count(True)
	accuracy = matches / n_validation
	print('nb accuracy:', 100 * accuracy, '%')

	x = knn_faces_features(faces_list)
	train_x = x[:-n_validation]
	validation_x = x[-n_validation:]
	knn_model = KNN()
	knn_model.train(train_x, train_y)
	matches = list(knn_model.predict(validation_x) == validation_y).count(True)
	accuracy = matches / n_validation
	print('KNN accuracy:', 100 * accuracy, '%')
