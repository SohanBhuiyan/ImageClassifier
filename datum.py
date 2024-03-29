# samples.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import features
DATUM_WIDTH = 0  # in pixels
DATUM_HEIGHT = 0  # in pixels

class Datum:
    """
    A datum is a pixel-level encoding of digits or face/non-face edge maps.

    Digits are from the MNIST dataset and face images are from the
    easy-faces and background categories of the Caltech 101 dataset.


    Each digit is 28x28 pixels, and each face/non-face image is 60x74
    pixels, each pixel can take the following values:
      0: no edge (blank)
      1: gray pixel (+) [used for digits only]
      2: edge [for face] or black pixel [for digit] (#)

    Pixel data is stored in the 2-dimensional array pixels, which
    maps to pixels on a plane according to standard euclidean axes
    with the first dimension denoting the horizontal and the second
    the vertical coordinate:

      28 # # # #      #  #
      27 # # # #      #  #
       .
       .
       .
       3 # # + #      #  #
       2 # # # #      #  #
       1 # # # #      #  #
       0 # # # #      #  #
         0 1 2 3 ... 27 28

    For example, the + in the above diagram is stored in pixels[2][3], or
    more generally pixels[column][row].

    The contents of the representation can be accessed directly
    via the getPixel and getPixels methods.
    """

    def __init__(self, data, width, height):
        """
        Create a new datum from file input (standard MNIST encoding).
        """
        DATUM_HEIGHT = height
        DATUM_WIDTH = width
        self.height = DATUM_HEIGHT
        self.width = DATUM_WIDTH
        if data == None: # creates a empty array
            data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
        else:
            self.pixels = util.convertImageToNumpyArray(data)

    def getPixel(self, column, row):
        """
        Returns the value of the pixel at column, row as 0, or 1.
        """
        return self.pixels[column][row]

    def getPixels(self):
        """
        Returns all pixels as a list of lists.
        """
        return self.pixels

    def getAsciiString(self):
        """
        Renders the data item as an ascii image.
        """
        rows = []
        data = self.pixels
        for row in data:
            ascii = map(util.asciiGrayscaleConversionFunction, row)
            rows.append("".join(ascii))
        return "\n".join(rows)

    def __str__(self):
        return self.getAsciiString()

# Testing

def _test():
    n = 2
    items = util.loadDataFile("facedata/facedatatrain", n,60,70)
    # labels = loadLabelsFile("facedata/facedatatrainlabels", n)
   # items = util.loadDataFile("digitdata/trainingimages", n, 28, 28)
    labels = util.loadLabelsFile("digitdata/traininglabels", n)
    datumItem = items[0]
    print(datumItem.getAsciiString())



    features.rawPixelFeature(datumItem.pixels)
if __name__ == "__main__":
    _test()
