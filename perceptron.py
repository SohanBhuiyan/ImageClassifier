"""
Define a perceptron model.
"""


import numpy as np
from model import Model


class Perceptron(Model):
	"""Perceptron model which can handle a varying number of features and multiple output labels."""
	
	def __init__(self, n_labels: int):
		self.n_labels = n_labels
		
	def train(self, x: np.array, y: np.array):
		raise NotImplementedError
	
	def predict(self, x: np.array) -> np.array:
		raise NotImplementedError