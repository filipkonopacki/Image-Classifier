"""
Class for managing features of images
"""


# Libraries
import cv2
import numpy as np


# My classes
from Log import log
from Utils import util, FEATURE_TYPES, MATRIX, AVG_SATURATION, UNIQUE_COLORS, NEW, NO, \
    SYNTHETIC, REAL, IMAGE_TYPES, CREATE, LOAD, OFFSETS_DICT


class FeaturesManager:

    def __init__(self, test):
        self.test = test
        self.features_vector = test.feature_vector
        self.test.feature_vector = []
        self.offsets = []
        self.width = None
        self.height = None

        if self.test.feature_vector == NEW:
            self.action_type = CREATE
        else:
            self.action_type = LOAD

    def count_different_colors_ratio(self, image):
        n_unique_pixels = len(np.unique(image))
        n_total_pixels = len(image)
        unique_colors_ratio = n_unique_pixels / n_total_pixels
        return unique_colors_ratio

    def count_saturation_average(self, image):
        saturations = []
        for x in range(self.width):
            for y in range(self.height):
                [r, g, b] = image[x, y]
                saturation = max(abs(int(r) - int(g)), abs(int(r) - int(b)), abs(int(g) - int(b)))
                saturations.append(saturation)
        saturation_ratio = sum(saturations) / self.width * self.height
        return saturation_ratio

    def count_cooccurrence_matrix(self, image, max_brightness=256, channel=0, offset=[1, 0]):
        channel = self.test.channels.index(channel)
        if channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            channel = 0
        elif channel == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            channel = 0

        if len(image.shape) > 2:
            image = image[:, :, channel]
        cooccurrence_matrix = np.zeros([max_brightness, max_brightness])

        for x in range(self.width - offset[0]):
            for y in range(self.height - offset[1]):
                row_index = image[x, y]
                column_index = image[x + offset[0], y + offset[1]]
                cooccurrence_matrix[row_index, column_index] += 1

        return cooccurrence_matrix

    def count_ratio(self, matrix):
        pixels_on_diagonal = 0
        all_pixels = self.width * self.height
        for index in range(256):
            pixels_on_diagonal += matrix[index, index]
        return pixels_on_diagonal / all_pixels

    def prepare_offsets(self):
        for offset in self.test.offsets:
            self.offsets.append(OFFSETS_DICT[offset])

    def create_feature_vector(self, test_image=None):
        if len(self.offsets) == 0 and MATRIX in self.test.features:
            self.prepare_offsets()
        image_name = None
        if test_image is not None:
            self.test.decision[test_image] = []
            image_name = test_image.split('\\')
            image_name = image_name[-1]
        image_types = IMAGE_TYPES if test_image is None else [image_name]
        for image_type in image_types:
            images = self.test.images[image_type] if test_image is None else [cv2.imread(test_image)]
            for image in images:
                image_name = image_type + '_' + str(util.index) if test_image is None else image_name
                log.info('Working on image: ' + image_type + '_' + str(util.index))
                self.width, self.height, channel = image.shape
                tmp_vetor = []
                for feature in self.test.features:
                    if feature == UNIQUE_COLORS:
                        unq_colors = self.count_different_colors_ratio(image)
                        tmp_vetor.append(unq_colors)
                    if feature == AVG_SATURATION:
                        avg_sat = self.count_saturation_average(image)
                        tmp_vetor.append(avg_sat)
                    if feature == MATRIX:
                        for offset in self.offsets:
                            for channel in self.test.channels:
                                matrix = self.count_cooccurrence_matrix(image, 256, channel, offset)
                                if self.test.save_options['matrices'] != NO:
                                    util.save_matrix(matrix, image_type, channel, offset,
                                                     self.test.save_options['matrices'])
                                ratio = self.count_ratio(matrix)
                                tmp_vetor.append(ratio)
                util.index = util.index + 1
                if test_image is None:
                    tmp_vetor.append(image_type)
                    self.test.feature_vector.append(tmp_vetor)
                else:
                    self.test.decision[test_image].append(tmp_vetor)
                log.info('Finished working on image {}_{}'.format(image_type, str(util.index)))
            util.index = 1
        if test_image is None:
            if self.test.save_options['vector'] != NO:
                util.save_vector(self.test)
            if self.test.save_options['histograms'] != NO:
                util.save_histograms(self.test)

    # if test_image is not None:
        #     if len(self.offsets) == 0:
        #         self.prepare_offsets()
        #     self.test.decision[test_image] = []
        #     image = cv2.imread(test_image)
        #     tmp_vetor = []
        #     self.width, self.height, channel = image.shape
        #     for feature in self.test.features:
        #         if feature == UNIQUE_COLORS:
        #             unq_colors = self.count_different_colors_ratio(image)
        #             tmp_vetor.append(unq_colors)
        #         if feature == AVG_SATURATION:
        #             avg_sat = self.count_saturation_average(image)
        #             tmp_vetor.append(avg_sat)
        #         if feature == MATRIX:
        #             for offset in self.offsets:
        #                 for channel in self.test.channels:
        #                     matrix = self.count_cooccurrence_matrix(image, 256, channel, offset)
        #                     ratio = self.count_ratio(matrix)
        #                     tmp_vetor.append(ratio)
        #     self.test.decision[test_image].append(tmp_vetor)
        #     return
        # else:

    def get_features_vector(self, test_image=None):
        if test_image is not None:
            log.info('Making features vector for test image.')
            self.create_feature_vector(test_image)
            log.info('Done')
            return
        if self.features_vector != NEW:
            log.info('Loading features vector from file {} .'.format(self.features_vector))
            util.read_txt(self.test, self.features_vector)
            if self.test.save_options['histograms'] != NO:
                util.save_histograms(self.test)
            log.info('Done')
        else:
            self.create_feature_vector()
