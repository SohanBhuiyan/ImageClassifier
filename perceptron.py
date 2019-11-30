"""
Define a perceptron model.
"""


import numpy as np
from model import Model


class Perceptron(Model):
	"""
	Perceptron model which can handle a varying number of features and multiple output labels.
	Note: this doesn't use a bias because the Berkeley version doesn't, but we might add bias later.
	"""

	def train(self, x: np.array, y: np.array, n_epochs=5):
		n_features = x.shape[1]
		n_labels = y.shape[1]
		self.weights = np.random.rand(n_features, n_labels)

		for _ in range(n_epochs):
			prediction = self.predict(x)
			for features, predicted_labeling, actual_labeling in zip(x, prediction, y):
				predicted_label = predicted_labeling.argmax()
				actual_label = actual_labeling.argmax()
				if predicted_label != actual_label:
					# we got it wrong, so we have to correct it
					self.weights[:, actual_label] += features
					self.weights[:, predicted_label] -= features

	def predict(self, x: np.array) -> np.array:
		# instead of multiplying each row by weight vectors,
		# we can just use a matrix multiplication to do all of this at once.
		# unforetunately, we have to briefly convert to matrices to do this.
		return np.array(np.matrix(x) * np.matrix(self.weights))


if __name__ == '__main__':
	# we create an example perceptron.
	# our perceptron will read a vector and label whether the mean is positive or negative.
	# it is a 2-class perceptron (pos, neg), so (1, 0) means that it's positive and (0, 1) means that it's negative.

	# let's pretend that we can take a sample of 10 values from a uniform distribution over [-0.5, 0.5],
	# so we take 1000 samples and hold each as a row in a matrix.
	# we subtract by 0.5 because np.random.rand samples over Unif(0, 1)
	n_numbers_per_sample = 3
	n_train_samples = 10000
	n_validation_samples = 500
	train_x = np.random.rand(n_train_samples, n_numbers_per_sample) - 0.5
	train_y = np.array([[1, 0] if row.mean() >= 0 else [0, 1] for row in train_x], dtype=np.float64)
	validation_x = np.random.rand(n_validation_samples, n_numbers_per_sample) - 0.5
	validation_y = np.array([[1, 0] if row.mean() >= 0 else [0, 1] for row in validation_x], dtype=np.float64)


	print('Training input (beginning):')
	print(train_x[:3])
	print('Training labels (beginning):')
	print(train_y[:3])

	# now, we can make the model
	model = Perceptron()
	model.train(train_x, train_y)

	print('Classifications (beginning):')
	print(validation_x[:10])
	print(model.predict(validation_x).argmax(axis=1)[:10])

	inaccuracy = np.sum(np.abs((model.predict(validation_x).argmax(axis=1) - validation_y.argmax(axis=1)))) / n_validation_samples
	print('Accuracy:', 100 * (1 - inaccuracy), '%')

	print('Weights:')
	print(model.weights)


	# at this point, we can generate some statistics
	print()
	for n_epochs in range(5):
		accuracies = []
		for _ in range(10):
			train_x = np.random.rand(n_train_samples, n_numbers_per_sample) - 0.5
			train_y = np.array([[1, 0] if row.mean() >= 0 else [0, 1] for row in train_x], dtype=np.float64)
			validation_x = np.random.rand(n_validation_samples, n_numbers_per_sample) - 0.5
			validation_y = np.array([[1, 0] if row.mean() >= 0 else [0, 1] for row in validation_x], dtype=np.float64)

			model = Perceptron()
			model.train(train_x, train_y, n_epochs=n_epochs)

			inaccuracy = np.sum(np.abs((model.predict(validation_x).argmax(axis=1) - validation_y.argmax(axis=1)))) / n_validation_samples
			accuracies.append(100 * (1 - inaccuracy))
		print('Mean accuracy with', n_epochs, 'epochs:', sum(accuracies) / len(accuracies))
