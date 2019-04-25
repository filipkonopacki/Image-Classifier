"""
# 
"""


# Libraries
import pandas as pd
from sklearn import linear_model, metrics, svm
from sklearn.model_selection import train_test_split


# My classes
from Log import log
from Utils import AVG_SATURATION, MATRIX, UNIQUE_COLORS, FEATURE_TYPES, SVC


class Classifiers:

    def __init__(self, test):
        self.test = test
        self.labels = []
        self.data_set = None
        self.model = None

    def __set_labels(self):
        for feature in self.test.features:
            if feature == MATRIX:
                for offset in self.test.offsets:
                    for channel in self.test.channels:
                        self.labels.append(channel + offset)
            else:
                self.labels.append(feature)
        self.labels.append('ImageType')

    def setup(self):
        self.__set_labels()
        self.data_set = pd.DataFrame(self.test.feature_vector, columns=self.labels)

    def train(self):
        for prediction in self.test.prediction_set:
            try:
                percentage = int(prediction)
                test_size = 100 - percentage
                log.info('Split data: ' + str(percentage) + '% of training data and ' + str(
                    test_size) + '% of test data')
                test_size = test_size / 100
            except:
                test_size = 1

            train_x, test_x, train_y, test_y = train_test_split(self.data_set[self.labels[:-1]],
                                                                self.data_set[self.labels[-1]],
                                                                test_size=test_size)
            log.info('Training model...')
            self.model = svm.SVC() if self.test.classifier == SVC else linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
            self.model.fit(train_x, train_y)

            if test_size != 1:
                index = str(prediction)
                self.test.decision[index] = self.test.classifier + ' accuracy for train set: ' + \
                                     str(round(metrics.accuracy_score(train_y, self.model.predict(train_x)), 2)) + '\t' + \
                                     self.test.classifier + ' accuracy for test set: ' + \
                                     str(round(metrics.accuracy_score(test_y, self.model.predict(test_x)), 2))

                log.result(self.test.decision[prediction])
            else:
                print(self.test.decision[prediction])
                decision = self.model.predict(self.test.decision[prediction])
                self.test.decision[prediction] = str(decision)
                log.result(self.test.classifier + ' decision on passed image is: ' + self.test.decision[prediction])

    def run(self):
        self.setup()
        self.train()
