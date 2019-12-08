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


def multipleGridFeatures(matrix):
    row = len(matrix)
    col = len(matrix[0])
    vector = util.createVector(row * col)
    vIndex = 0

    startingx = 5
    startingy = 5
    temp = 0

    for i in range(row):
        for j in range(col):
            if (i < 5 or j < 5):
                continue
            else:
                for k in range(5):
                    for l in range(5):
                        # print(i+k)
                        # print(j+l)
                        if not((i + k) >= row or (j + l) >= col):
                            temp += matrix[i+k][j+l]

                if temp > 15:
                    vector[vIndex] = 1
                    vIndex = vIndex + 1
                    temp = 0
                else:
                    vector[vIndex] = 0
                    vIndex = vIndex + 1
                    temp = 0
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

def imageCompressor(matrix):
    row = len(matrix)
    col = len(matrix[0])
    vector = util.createVector(row * col)
    vIndex = 0

    dim = 2

    temp = np.zeros((int(row/dim), int(col/dim)))

    # print(matrix)
    #matrix = np.array(matrix)
    for i in range(0,row, dim):
        for j in range(0,col, dim):
            # print(i, j)
            for x in range(0, dim):
                for y in range(0, dim):
                    if not ((i + x) >= row or (j + y) >= col):
                        if int((i + x)/dim) < int(row/dim) and int((j + y)/dim) < int(col/dim):
                            #temp[int((i + x)/dim)][int((j + y)/dim)] += matrix[(i + x)][(j + y)]
                            temp[int((i + x) / dim)][int((j + y) / dim)] = max(temp[int((i + x)/dim)][int((j + y)/dim)],matrix[(i + x)][(j + y)])

    vector = np.ndarray.flatten(temp)
    # print(vector)
    #print(temp)
    return vector
