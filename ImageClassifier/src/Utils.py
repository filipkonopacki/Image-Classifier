"""
# Class for storing utilities used in image classification
"""


# Libraries
import os
import cv2
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt

# My classes
from Log import log


# Constants
##################################################################
MATRIX = 'cooccurrence_matrices'
AVG_SATURATION = 'avg_saturation'
UNIQUE_COLORS = 'unique_colors'
FEATURE_TYPES = [MATRIX, AVG_SATURATION, UNIQUE_COLORS]

SYNTHETIC = 'synthetic'
REAL = 'real'
IMAGE_TYPES = [SYNTHETIC, REAL]

NEW = 'new'
NO = 'no'

OFFSETS = ['10', '20', '40', '80', '01', '02', '04', '08']
CHANNELS = ['R', 'G', 'B', 'H', 'GRAY']

LOGISTIC_REGRESSION = 'logistic_regression'
SVC = 'SVC'
CLASSIFIERS = [LOGISTIC_REGRESSION, SVC]

CREATE = 'create'
LOAD = 'load'

OFFSETS_DICT = {'10': [1, 0],
           '20': [2, 0],
           '40': [4, 0],
           '80': [8, 0],
           '01': [0, 1],
           '02': [0, 2],
           '04': [0, 4],
           '08': [0, 8]}

##################################################################


class Utils:

    def __init__(self):
        self.matrix_folder_path = None
        self.txt_path = None
        self.histogram_folder_path = None
        self.index = 1

    def __create_inner_folders(self, root, outer_folder, image_type):
        outer_path = root + '\\' + str(outer_folder) + '\\'
        if not os.path.exists(outer_path):
            os.mkdir(outer_path)
            log.info('Folder created at: ' + outer_path)
        inner_path = outer_path + image_type
        if not os.path.exists(inner_path):
            os.mkdir(inner_path)
            log.info('Folder created at: ' + inner_path)
        return inner_path

    # returns vector with features separated by the type of feature and offset e.g. one row are ratios
    # for channel R and offset [1, 0]
    def __get_feature_vector(self, features_vector):
        tmp_features = []
        for i in range(len(features_vector[0]) - 1):
            tmp_vector = []
            for j in range(len(features_vector)):
                tmp_vector.append(features_vector[j][i])
            tmp_features.append(tmp_vector)
        return tmp_features

    def prepare_data(self, real_features_vector, synthetic_features_vector ):
        data = defaultdict(list)
        data[REAL] = self.__get_feature_vector(real_features_vector)
        data[SYNTHETIC] = self.__get_feature_vector(synthetic_features_vector)
        return data

    def __draw_histogram(self, data, image_type, test):
        ylabel = 'liczba obrazów'
        if len(test.features) == 1 and test.features[0] == MATRIX:
            xlabel = 'liczba pikseli na przekątnej macierzy współwystępowania / liczba wszystkich pikseli'
            offset_index = 0
            channel_index = 0
            for i in range(len(data)):
                if channel_index >= len(test.channels):
                    channel_index = 0
                    offset_index = offset_index + 1
                channel = test.channels[channel_index]
                offset = test.offsets[offset_index]
                inner_folder = 'offset[' + offset + ']'
                image_type_path = self.__create_inner_folders(self.histogram_folder_path, inner_folder, image_type)
                hist_data = list(Counter(data[i]))
                plt.hist(hist_data, bins=20)
                title = 'Histogram dla kanału ' + channel
                file_name = 'histogram_' + channel
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                file_path = image_type_path + '\\' + file_name + '.png'
                try:
                    plt.savefig(file_path)
                    plt.close()
                    log.info('Saving histogram in ' + file_path)
                except:
                    log.error('Saving histogram in ' + file_path + ' failed')
                finally:
                    channel_index = channel_index + 1
        else:
            for i in range(len(test.features)):
                image_type_path = self.__create_inner_folders(self.histogram_folder_path, test.features[i], image_type)
                title = 'Histogram średniego nasycenia' if test.features[i] == AVG_SATURATION else 'Histogram unikalnych kolorów'
                xlabel = 'średnie nasycenie' if test.features[i] == AVG_SATURATION else 'liczba pikseli o unikalnym kolorze / liczba wszystkich pikseli'
                file_name = 'histogram_' + test.features[i]
                hist_data = list(Counter(data[i]))
                plt.hist(hist_data, bins=20)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                file_path = image_type_path + '\\' + file_name + '.png'
                try:
                    plt.savefig(file_path)
                    plt.close()
                    log.info('Saving histogram in ' + file_path)
                except:
                    log.error('Saving histogram in ' + file_path + ' failed')

    def split_features_vector(self, features_vector):
        real_features_vector = []
        synthetic_features_vector = []
        for line in features_vector:
            if line[-1] == SYNTHETIC:
                synthetic_features_vector.append(line)
            else:
                real_features_vector.append(line)
        return real_features_vector, synthetic_features_vector

    def save_histograms(self, test):
        self.histogram_folder_path = test.save_options['histograms']
        log.info('Preparing data...')
        real_features_vector, synthetic_features_vector = self.split_features_vector(test.feature_vector)
        data = self.__prepare_data(real_features_vector, synthetic_features_vector)
        for image_type in IMAGE_TYPES:
            self.__draw_histogram(data[image_type], image_type, test)

    def save_matrix(self, matrix, image_type, channel, offset, path):
        self.matrix_folder_path = path
        image_type_path = self.__create_inner_folders(self.matrix_folder_path, offset, image_type)
        file_path = image_type_path + '\\matrix_' + str(self.index) + '__' + channel + '.jpg'
        cv2.imwrite(file_path, matrix)
        if os.path.exists(file_path):
            log.info('Image saved in ' + file_path)
        else:
            log.error('Failed to save image in ' + file_path)

    def save_vector(self, test):
        try:
            file = open(test.save_options['vector'], 'w')
        except:
            log.error('Entered path is incorrect.')
            raise NameError('Incorrect input')

        for line in test.feature_vector:
            for element in line:
                file.write(str(element) + ';')
            file.write('\n')
        file.close()
        log.info('Features vector saved in ' + test.save_options['vector'])

    def read_txt(self, test, path):
        file = open(path, 'r')
        for line in file:
            tmp_vector = []
            elements = line.split(';')
            for element in elements:
                if element not in IMAGE_TYPES:
                    try:
                        element = float(element)
                    except:
                        continue
                tmp_vector.append(element)
            test.feature_vector.append(tmp_vector)


util = Utils()
