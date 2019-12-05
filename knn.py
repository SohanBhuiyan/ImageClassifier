"""
Define a decision tree model.
"""


import numpy as np
import pandas as pd
import random
from model import Model


class KNN(Model):
	"""
	This is an implementation of the kNN technique based on our textbook's description.
	It intakes a matrix of item's features.
	The output is a one-hot vector with as many components as there are classes.
	I'm still implementing this.
	"""

	def __init__(self, k: int=5):
		"""
		All the options to kNN are defined here.
		k: the number of neighbours to consider when finding the class
		"""

		assert k % 2 == 1
		self.k = k

	def train(self, x: np.array, y: np.array):
		self.train_x = x
		self.train_y = y
		self.n_classes = len(set(self.train_y))

		self.mins = x.min(axis=0)
		self.maxes = x.max(axis=0)

	def predict(self, x: np.array) -> np.array:
		predictions = []

		for i, item in enumerate(x):
			# we find how close each point is using Manhattan distance,
			# but we can change this later
			distance = lambda a, b: np.sum(np.abs(a - b), axis=1)
			normalize = lambda u: (u - self.mins) / (self.maxes - self.mins)
			distances = distance(normalize(self.train_x), normalize(item))
			points_sorted_by_distance = [u[0] for u in sorted(list(enumerate(distances)), key=lambda x:x[1])]
			neighbors = points_sorted_by_distance[:self.k]
			plurality = pd.Series(self.train_y[neighbors]).value_counts().keys()[0]
			predictions.append(plurality)

		return np.array(predictions)


if __name__ == '__main__':
	# we create an example decision tree
	# to track progress, we create dummy data


	# we make dummy data representing different houses in different cities.
	# in our dataset, we imagine that we have houses in three different cities,
	# and each house has three aspects: price, size, and altitude.
	# we make a kNN-based model that guesses where a given house is.

	n_samples_per_city = 1000
	nyc = np.array([
		np.random.rand(n_samples_per_city) * 100000000 + 1000000,
		np.random.rand(n_samples_per_city) * 10000 + 100,
		np.random.rand(n_samples_per_city) * 10,
	]).T
	sf = np.array([
		np.random.rand(n_samples_per_city) * 100000000 + 5000000,
		np.random.rand(n_samples_per_city) * 1000 + 100,
		np.random.rand(n_samples_per_city) * 100 + 20,
	]).T
	houston = np.array([
		np.random.rand(n_samples_per_city) * 1000000 + 10000,
		np.random.rand(n_samples_per_city) * 30000 + 3000,
		np.random.rand(n_samples_per_city) * 10,
	]).T


	n_validation = 100
	indices = list(range(3 * n_samples_per_city))
	random.shuffle(indices)
	validation = indices[:n_validation]
	not_validation = list(set(range(3 * n_samples_per_city)) - set(validation))
	x = np.concatenate([nyc, sf, houston])
	y = np.array(sorted([0, 1, 2] * n_samples_per_city))
	train_x = x[not_validation]
	train_y = y[not_validation]
	validation_x = x[validation]
	validation_y = y[validation]


	print('Training input (beginning):')
	print(train_x[:3])
	print('Training labels (beginning):')
	print(train_y[:3])

	# now, we can make the model
	model = KNN()
	model.train(train_x, train_y)

	print('Classifications inputs (beginning):')
	print(validation_x[:10])
	print('Classifications outputs (beginning):')
	print(model.predict(validation_x)[:10])
	print('Ideal classifications (beginning):')
	print(validation_y[:10])

	matches = list(model.predict(validation_x) == validation_y).count(True)
	accuracy = matches / n_validation
	print('Accuracy:', 100 * accuracy, '%')
