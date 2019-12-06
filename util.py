# util.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html



import zipfile
import os
import random
import numpy as np
from datum import Datum
def readlines(filename):
    "Opens a file or reads it from the zip archive data.zip"
    if (os.path.exists(filename)):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n')


def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


def asciiGrayscaleConversionFunction(value):
    """
    Helper function for display purposes.
    """
    if (value == 0):
        return ' '
    elif (value == 1):
        return '+'
    elif (value == 2):
        return '#'


def IntegerConversionFunction(character):
    """
    Helper function for file reading.
    """
    if (character == ' '):
        return 0
    elif (character == '+'):
        return 1
    elif (character == '#'):
        return 2

def isColoredPixel(pixelValue):
    if pixelValue == 0:  # 0 represents blank space
        return False
    else:
        return True

def convertToInteger(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            character = data[i][j]
            data[i][j] = IntegerConversionFunction(character)
    return data

def loadDataFile(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.
    (Return less then n items if the end of file is encountered).
    """
    DATUM_WIDTH = width
    DATUM_HEIGHT = height
    fin = readlines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < DATUM_WIDTH - 1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Datum(data, DATUM_WIDTH, DATUM_HEIGHT))
    return items

def convertImageToNumpyArray(image_matrix):
    integer_array = convertToInteger(image_matrix)
    numpy_array = np.asarray(integer_array)
    return numpy_array


def flipCoin(p):
    r = random.random()
    return r < p

# returns a numpy vector filled with 0s of size n
def createVector(n):
    vector = np.zeros(n)
    return vector



def splitFaceArray(array):
	return np.array_split(array,7)

def splitDigitArray(array):
    return np.array_split(array,7)




