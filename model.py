"""
Define a structure for what a model does so that the perceptron and naive bayes models have the same API.
"""


import numpy as np


def Model:
	"""Structure for a machine learning model"""
	
	def train(x: np.array, y: np.array):
		pass
	
	def predict(x: np.array) -> np.array:
		"""Read in the features matrix and return a matrix where 
		   the (i, j)th component is the likelihood of the ith datum
		   having label j."""
		
		pass