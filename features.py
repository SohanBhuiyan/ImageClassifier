import util
import numpy as np
"treat each pixel as its own feature"
def rawPixelFeature(matrix):
    row = len(matrix)
    col = len(matrix[0])
    vector = util.createVector(row * col)
    vIndex = 0

    for i in range(row):
        for j in range(col):
            vector[vIndex] = matrix[i][j]
            vIndex = vIndex+1

    return vector


# breaks a nxn matrix into 7 mxm segments
def gridFeature(matrix):
    split_matrices = util.splitFaceArray(matrix)
    grid_features = util.createVector(len(split_matrices))

    for i in range(len(split_matrices)):
        total_pixel_val = 0
        matrix = split_matrices[i] # matrix will represent 1 of 7 segments of split_matrices
        for pixel_val in np.nditer(matrix): # goes through each element in matrix
            total_pixel_val = total_pixel_val + pixel_val

        grid_features[i] = total_pixel_val

    return grid_features



def rowPixelFeature(matrix):
    row = len(matrix)
    col = len(matrix[0])
    vector = util.createVector(row)

    for i in range(row):
        coloredPixels = 0 # every row starts in the beginning
        for j in range(col):
            if util.isColoredPixel(matrix[i][j]):
                coloredPixels = coloredPixels + 1

        vector[i] = coloredPixels
    return vector

