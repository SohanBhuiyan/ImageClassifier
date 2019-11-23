import util

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



def gridFeature(matrix):
    None