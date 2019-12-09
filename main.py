"""
Read in the data and evaluate the models
"""
import time

import numpy as np
import util
from nb import NaiveBayes
from perceptron import Perceptron
from knn import KNN
from features import rawPixelFeature, rowPixelFeature, gridFeature, multipleGridFeatures, imageCompressor


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

def digitClassification(percent: int):
	n_total_digits = 450
	n_samples_digits = int(n_total_digits * (percent/100))
	n_testing_digits = 1000
	digit_list = read_digits_file("digitdata/trainingimages", n_samples_digits)
	digit_test_list = read_digits_file("digitdata/testimages", n_testing_digits)

	y = np.array(util.loadLabelsFile("digitdata/traininglabels", n_samples_digits))


	train_y = y
	test_y = np.array(util.loadLabelsFile("digitdata/testlabels", n_testing_digits))

	x = perceptron_faces_features(digit_list)
	train_x = x
	test_x = perceptron_faces_features(digit_test_list)
	p_model = Perceptron(100)

	start_time = time.time()
	p_model.train(train_x, train_y)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for training")

	start_time = time.time()
	matches = list(p_model.predict(test_x) == test_y).count(True)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for predicting")

	accuracy = matches / n_testing_digits
	print('Perceptron accuracy:', 100 * accuracy, '%')
	print()

	x = nb_faces_features(digit_list)
	train_x = x
	test_x = nb_faces_features(digit_test_list)
	nb_model = NaiveBayes()

	start_time = time.time()
	nb_model.train(train_x, train_y)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for training")

	start_time = time.time()
	matches = list(nb_model.predict(test_x) == test_y).count(True)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for predicting")

	accuracy = matches / n_testing_digits
	print('nb accuracy:', 100 * accuracy, '%')
	print()

	x = knn_faces_features(digit_list)
	train_x = x
	test_x = knn_faces_features(digit_test_list)
	knn_model = KNN()

	start_time = time.time()
	knn_model.train(train_x, train_y)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for training")

	start_time = time.time()
	matches = list(knn_model.predict(test_x) == test_y).count(True)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for predicting")

	accuracy = matches / n_testing_digits
	print('KNN accuracy:', 100 * accuracy, '%')
	print()

def faceClassification(percent: int):
	n_total_faces = 450
	n_samples_faces = int(n_total_faces * (percent / 100))
	n_testing_digits = 150
	face_list = read_faces_file("facedata/facedatatrain", n_samples_faces)
	face_test_list = read_faces_file("facedata/facedatatest", n_testing_digits)

	y = np.array(util.loadLabelsFile("facedata/facedatatrainlabels", n_samples_faces))

	train_y = y
	test_y = np.array(util.loadLabelsFile("facedata/facedatatestlabels", n_testing_digits))
	x = perceptron_faces_features(face_list)
	train_x = x
	test_x = perceptron_faces_features(face_test_list)
	p_model = Perceptron(100)

	start_time = time.time()
	p_model.train(train_x, train_y)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for training")

	start_time = time.time()
	matches = list(p_model.predict(test_x) == test_y).count(True)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for predicting")

	accuracy = matches / n_testing_digits
	print('Perceptron accuracy:', 100 * accuracy, '%')
	print()

	x = nb_faces_features(face_list)
	train_x = x
	test_x = nb_faces_features(face_test_list)
	nb_model = NaiveBayes()

	start_time = time.time()
	nb_model.train(train_x, train_y)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for training")

	start_time = time.time()
	matches = list(nb_model.predict(test_x) == test_y).count(True)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for predicting")

	accuracy = matches / n_testing_digits
	print('nb accuracy:', 100 * accuracy, '%')
	print()

	x = knn_faces_features(face_list)
	train_x = x
	test_x = knn_faces_features(face_test_list)
	knn_model = KNN()

	start_time = time.time()
	knn_model.train(train_x, train_y)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for training")

	start_time = time.time()
	matches = list(knn_model.predict(test_x) == test_y).count(True)
	elapsed_time = time.time() - start_time
	print('%.3f' % (elapsed_time) + " seconds for predicting")

	accuracy = matches / n_testing_digits
	print('KNN accuracy:', 100 * accuracy, '%')
	print()

if __name__ == '__main__':#demonstration

	print("Digit classification")
	print()
	total_start_time = time.time()
	for x in range(10, 101, 10):
		print(str(x) + '%' + ' of training data.')
		start_time = time.time()

		digitClassification(x)
		elapsed_time = time.time() - start_time
		print('%.3f'%(elapsed_time) + " seconds")
		print()

	print("Face or not face classification")
	print()
	for x in range(10, 101, 10):
		print(str(x) + '%' + ' of training data.')
		start_time = time.time()

		faceClassification(x)
		elapsed_time = time.time() - start_time
		print('%.3f'%(elapsed_time) + " seconds")
		print()

	elapsed_time = time.time() - total_start_time
	print("TOTAL TIME "+'%.3f'%(elapsed_time) + " seconds")


	# n_samples = 450
	# n_validation = n_samples // 10
	# faces_list = read_faces_file("facedata/facedatatrain", n_samples)
	#
	# y = np.array(util.loadLabelsFile("facedata/facedatatrainlabels", n_samples))
	# train_y = y[:-n_validation]
	# validation_y = y[-n_validation:]
	#
	# x = perceptron_faces_features(faces_list)
	# train_x = x[:-n_validation]
	# validation_x = x[-n_validation:]
	# p_model = Perceptron(100)
	# p_model.train(train_x, train_y)
	# matches = list(p_model.predict(validation_x) == validation_y).count(True)
	# accuracy = matches / n_validation
	# print('Perceptron accuracy:', 100 * accuracy, '%')
	#
	# x = nb_faces_features(faces_list)
	# train_x = x[:-n_validation]
	# validation_x = x[-n_validation:]
	# nb_model = NaiveBayes()
	# nb_model.train(train_x, train_y)
	# matches = list(nb_model.predict(validation_x) == validation_y).count(True)
	# accuracy = matches / n_validation
	# print('nb accuracy:', 100 * accuracy, '%')
	#
	# x = knn_faces_features(faces_list)
	# train_x = x[:-n_validation]
	# validation_x = x[-n_validation:]
	# knn_model = KNN()
	# knn_model.train(train_x, train_y)
	# matches = list(knn_model.predict(validation_x) == validation_y).count(True)
	# accuracy = matches / n_validation
	# print('KNN accuracy:', 100 * accuracy, '%')
