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


# breaks a nxn matrix into 9 mxm segments
def gridFeature(matrix):
    pass


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

