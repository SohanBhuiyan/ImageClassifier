import numpy as np
from model import Model


class NaiveBayes(Model):
    # x is a matrix of images and their features
    # y is the label of the image
    def train(self, x: np.array, y: np.array):
        numberOfFeatures = x.shape[1]
        numberOfLabels = y.shape[1]
        numberofImages = x.shape[0]

        self.table = [[dict() for y in range(numberOfFeatures)] for x in range(numberOfLabels)]
        self.labelProbability = self.unique(y)
        self.featureProbability = self.labels(x)
        self.numberOfImages = numberofImages


        #print(type(y[0]))
        labels, counts = np.unique(y, return_counts=True)
        #print(labels, counts, 'asdasd')
        #print(numberofImages, numberOfFeatures, numberOfLabels)

        for features, label in zip(x, y):
            # print(features, label)
            for x, y in enumerate(label):
                if y == 1:
                    # print(self.table[x])
                    # self.table[x]
                    # print("adsasdasdas", features[1])

                    for i in range(len(features)):
                        feat = features[i]
                        if feat in self.table[x][i]:
                            self.table[x][i][feat] += 1
                        else:
                            self.table[x][i][feat] = 1

                        if feat in self.featureProbability:
                            self.featureProbability[feat] += 1
                        else:
                            self.featureProbability[feat] = 1

        # print(len(self.table[0][0]), self.table[0][0])
        # print(self.table)
        # print(self.featureProbability)

                # print(x,y)
            # break


    def predict(self, x: np.array) -> np.array:
        #result = np.array(len(self.labelProbability))
        result = np.zeros(shape=(len(self.labelProbability)))
        imageGivenLabel = 1


        for label in range(len(self.labelProbability)):
            #print(label, "PREDICT")
            for num, y in enumerate(self.table[label]):
                #print(num, y, "ANOTHER")
                if x[num] in y:
                    #print(imageGivenLabel)
                    imageGivenLabel *= (y[x[num]] /self.labelProbability[label])
                else:
                    imageGivenLabel *= 0.001
            equation = (1/self.numberOfImages) * imageGivenLabel * (self.labelProbability[label] / self.numberOfImages)
            result[label] = equation
        return result

    def unique(self, y: np.array) -> dict:
        labelDictionary = dict()
        for labels in y:
            #print(labels, "eee", labels[0])
            for num,label in enumerate(labels):
                #print(label, num)
                if label == 1:
                    if num in labelDictionary:
                        labelDictionary[num] += 1
                    else:
                        labelDictionary[num] = 1

        return labelDictionary


    def labels(self, x: np.array) -> dict:
        featureDictionary = dict()

        for feats in x:
            for y in feats:
                if y in featureDictionary:
                    featureDictionary[y] += 1
                else:
                    featureDictionary[y] = 1

        return featureDictionary


if __name__ == '__main__':
    n_numbers_per_sample = 3
    n_train_samples = 10000
    n_validation_samples = 500
    train_x = np.random.rand(n_train_samples, n_numbers_per_sample) - 0.5
    train_y = np.array([[1, 0] if row.mean() >= 0 else [0, 1] for row in train_x], dtype=np.float64)
    validation_x = np.random.rand(n_validation_samples, n_numbers_per_sample) - 0.5
    validation_y = np.array([[1, 0] if row.mean() >= 0 else [0, 1] for row in validation_x], dtype=np.float64)

    train_x = np.round(train_x, 1)
    print('Training input (beginning):')
    print(train_x[:3])
    print('Training labels (beginning):')
    print(train_y[:3])
    validation_x = np.round(validation_x, 1)
    # now, we can make the model
    print('train')
    model = NaiveBayes()
    print(validation_x[1], "VALID", validation_y[1])
    model.train(train_x, train_y)
    
    temp = 0
    for w in range(len(validation_x)):
        if np.argmax(validation_y[w]) == np.argmax(model.predict(validation_x[w])):
            # print(validation_y[w])
            temp += 1

    print(temp/10000)

