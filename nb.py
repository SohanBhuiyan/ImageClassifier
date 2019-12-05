import math
import numpy as np
import random
from functools import reduce
from model import Model


prod = lambda items: reduce(lambda a, b: a * b, items, 1)


class NaiveBayes(Model):
	"""
	This model assumes that we only use Boolean features.
	"""

	# note that this class internally doesn't use Numpy array since we don't do any linear algebra operations

	def train(self, x: np.array, y: np.array):
		self.train_x = list(x)
		self.train_y = list(y)
		self.labels = sorted(list(set(list(y))))
		features = list(range(x.shape[1]))

		self._probability = {label: self.train_y.count(label) / len(self.train_y) for label in self.labels}
		self._probability_given = {
			(feature, label): [[x[feature]].count(1) for i, x in enumerate(self.train_x) if self.train_y[i] == label].count(True) / self.train_y.count(label)
			for feature in features
			for label in self.labels}

	def predict(self, x: np.array) -> np.array:
		return np.array([
			np.argmax([
				self._probability[label] * prod([self._probability_given[i, label] if feature else 1 - self._probability_given[i, label] for i, feature in enumerate(item)])
				for label in self.labels])
			for item in x])


if __name__ == '__main__':
	# to test the model, we want to give it some dummy data.
	# specifically, we will give a bunch of boolean attributes about a person
	# there is also a third class, which is for dogs

	n_samples_per_class = 10000

	man = lambda:   [random.random() < 0.8, random.random() < 0.1, random.random() < 0.3]
	woman = lambda: [random.random() < 0.1, random.random() < 0.9, random.random() < 0.1]
	dog = lambda:   [random.random() < 0.2, random.random() < 0.0, random.random() < 0.8]
	men = np.array([man() for _ in range(n_samples_per_class)])
	women = np.array([woman() for _ in range(n_samples_per_class)])
	dogs = np.array([dog() for _ in range(n_samples_per_class)])
	x = np.concatenate([men, women, dogs]).astype(np.int8)
	y = np.array(sorted([0, 1, 2] * (n_samples_per_class)))
	n_validation = n_samples_per_class // 10
	indices = list(range(3 * n_samples_per_class))
	random.shuffle(indices)
	validation = indices[:n_validation]
	not_validation = list(set(range(3 * n_samples_per_class)) - set(validation))
	train_x = x[not_validation]
	train_y = y[not_validation]
	validation_x = x[validation]
	validation_y = y[validation]

	print('Training input (beginning):')
	print(train_x[:3])
	print('Training labels (beginning):')
	print(train_y[:3])
	validation_x = np.round(validation_x, 1)

	# now, we can make the model
	model = NaiveBayes()
	model.train(train_x, train_y)

	n_correct = 0

	accuracy = list(model.predict(validation_x) == validation_y).count(True) / n_validation

	print('Accuracy:', accuracy)
