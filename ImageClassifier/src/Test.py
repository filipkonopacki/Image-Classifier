"""
# Class representing and running test suites
"""

# Libraries
import datetime
import os
import cv2

# My classes
from Log import log, ROOT_DIR
from Utils import FEATURE_TYPES, NEW, MATRIX, LOGISTIC_REGRESSION, OFFSETS, CHANNELS, CLASSIFIERS


class Test:

    def __init__(self, test_node=None, test_name=None):

        self.test_node = test_node
        self.test_name = test_name
        self.feature_vector = None
        self.feature_vector_path = None
        self.real_images_location = None
        self.synthetic_images_location = None
        self.images = {}
        self.features = []
        self.channels = None
        self.offsets = None
        self.save_options = {}
        self.classifier = None
        self.prediction_set = None
        self.result_file = None
        self.decision = {}

        self.save_options['histograms'] = 'no'
        self.save_options['matrices'] = 'no'
        self.save_options['vector'] = 'no'

    def __validate_input(self):
        log.info('Checking input...')
        if self.feature_vector != NEW:
            if not os.path.exists(self.feature_vector):
                log.error(
                    'Invalid features vector. Get \'{}\', expected \'new\' or path to txt file.'.format(self.feature_vector))
                raise NameError('Invalid input')
        if self.feature_vector == NEW:
            if not os.path.exists(self.real_images_location):
                log.error('Invalid real images location. Path \'{}\' doesn\'t exist.'.format(self.real_images_location))
                raise NameError('Invalid input')
            if not os.path.exists(self.synthetic_images_location):
                log.error('Invalid synthetic images location. Path \'{}\' doesn\'t exist.'.format(self.synthetic_images_location))
                raise NameError('Invalid input')
        for feature in self.features:
            if feature not in FEATURE_TYPES:
                log.error('Invalid feature type. Get \'{}\', expected values: '.format(feature) + str(FEATURE_TYPES))
                raise NameError('Invalid input')
            if feature == MATRIX:
                for channel in self.channels:
                    if channel not in CHANNELS:
                        log.error('Invalid image channel. Get \'{}\', expected values: '.format(channel) + str(CHANNELS))
                        raise NameError('Invalid input')
                for offset in self.offsets:
                    if offset not in OFFSETS:
                        log.error(
                            'Invalid offset. Get \'{}\', expected values: '.format(offset) + str(OFFSETS))
                        raise NameError('Invalid input')
        if self.classifier not in CLASSIFIERS:
            log.error('Invalid classifier. Get \'{}\', expected values: '.format(self.classifier) + str(CLASSIFIERS))
            raise NameError('Invalid input')
        for prediction in self.prediction_set:
            try:
                prediction = int(prediction)
                if prediction >= 100:
                    log.error('Invalid prediction set value. Get \'{}\', expected values under 100 '.format(self.prediction_set))
                    raise NameError('Invalid input')
            except:
                if not os.path.exists(prediction):
                    log.error('Invalid prediction set. Get \'{}\', expected int under 100 or path to image file.'.format(self.prediction_set))
                    raise NameError('Invalid input')
                pass

        log.info('Input correct!')

    def __parse_xml(self):
        self.feature_vector = self.test_node.getElementsByTagName('feature_vector')[0].childNodes[0].nodeValue
        if self.feature_vector == NEW:
            self.real_images_location = self.test_node.getElementsByTagName('real_images')[0].childNodes[0].nodeValue
            self.synthetic_images_location = self.test_node.getElementsByTagName('synthetic_images')[0].childNodes[0].nodeValue

        features = self.test_node.getElementsByTagName('features')[0]
        for feature in features.childNodes:
            try:
                name = feature.getElementsByTagName('name')[0].childNodes[0].nodeValue
                self.features.append(name)
                if name == MATRIX:
                    channels = feature.getElementsByTagName('channels')[0].childNodes[0].nodeValue
                    self.channels = channels.split(';')
                    offsets = feature.getElementsByTagName('offsets')[0].childNodes[0].nodeValue
                    self.offsets = offsets.split(';')
            except:
                pass

        self.save_options['vector'] = self.test_node.getElementsByTagName('vector')[0].childNodes[0].nodeValue
        self.save_options['histograms'] = self.test_node.getElementsByTagName('histograms')[0].childNodes[0].nodeValue
        self.save_options['matrices'] = self.test_node.getElementsByTagName('matrices')[0].childNodes[0].nodeValue
        self.classifier = self.test_node.getElementsByTagName('classifier')[0].childNodes[0].nodeValue
        self.prediction_set = self.test_node.getElementsByTagName('prediction_set')[0].childNodes[0].nodeValue
        self.prediction_set = self.prediction_set.split(';')

        self.__validate_input()

    def __get_images_location(self, images_type):
        log.request('Enter path to folder with ' + images_type + ': ')
        path = input('Path: ')
        log.input('user> ' + path)
        return path

    def __get_user_input(self):
        self.test_name = 'Test suite from user input'
        log.request('Enter \'new\' or path to existing features vector.')
        self.feature_vector = input('Feature vector: ')
        log.input('user> ' + self.feature_vector)
        self.real_images_location = self.__get_images_location('real images')
        self.synthetic_images_location = self.__get_images_location('synthetic images')
        log.request('Enter features you want to use in classification separated by semicolon (;).')
        features = input('Features: ')
        log.input('user> ' + features)
        self.features = features.split(';')
        print(self.features)
        for feature in self.features:
            if feature == MATRIX:
                log.request('Enter image channels you want to use in classification separated by semicolon (;).')
                channels = input('Channels: ')
                log.input('user> ' + channels)
                self.channels = channels.split(';')
                log.request('Enter offsets you want to use in classification separated by semicolon (;). Example: offset [1,0] and [2,0] will be \'10;20\'.')
                offsets = input('Offsets: ')
                log.input('user> ' + offsets)
                self.offsets = offsets.split(';')
                log.request('Do you want to save co-occurrence matrices? Enter \'no\' or path to file.')
                self.save_options['matrices'] = input('Matrices: ')
                log.input('user> ' + self.save_options['matrices'])
            else:
                self.save_options['matrices'] = 'no'
        log.request('Do you want to save histograms of features? Enter \'no\' or path to file.')
        self.save_options['histograms'] = input('Histograms: ')
        log.input('user> ' + self.save_options['histograms'])
        log.request('Do you want to save features vector? Enter \'no\' or path to file.')
        self.save_options['vector'] = input('Features vector: ')
        log.input('user> ' + self.save_options['vector'])
        log.request('Enter name of classifier you want to use.')
        self.classifier = input('Classifier: ')
        log.input('user> ' + self.classifier)
        log.request('Enter prediction set you want to use (percentage of training data or path to image you want to classify). ')
        self.prediction_set = input('Prediction set: ')
        log.input('user> ' + self.prediction_set)

        self.__validate_input()

    def __load_images(self, path):
        images = []
        for root, dirs, files in os.walk(path):
            for name in files:
                file = path + '\\' + name
                image = cv2.imread(file)
                if image is None:
                    log.error('File in following location doesn\'t exist or is incorrect (expected image) : ' + file)
                else:
                    width, height, depth = image.shape
                    if width > 750:
                        scale = 750 / width
                        width = 750
                        height = int(height * scale)
                        image = cv2.resize(image, (width, height))
                    elif height > 750:
                        scale = 750 / height
                        height = 750
                        width = int(width * scale)
                        image = cv2.resize(image, (width, height))
                    images.append(image)
        if len(images) == 0:
            log.error('No image was found.')
            raise NameError('No image found')

        return images

    def save_result(self, test_xml=None):
        log.info('Saving result...')
        result_dir = ROOT_DIR + '\\results'
        if not os.path.exists(result_dir):
            log.warning('Result directory doesn\'t exist. Creating new.')
            os.mkdir(result_dir)
        today = datetime.date.today()
        result_path = result_dir + '\\' + self.test_name + '__' + today.__str__() + '.txt'
        result_file = open(result_path, 'a')
        result_file.write('**********Result**********\n')
        time = datetime.datetime.now()
        time = time.strftime('%X')
        result_file.write('Finished: {} {}\n'.format(today.__str__(), time))
        if test_xml is not None:
            result_file.write('Test xml: {}\n'.format(test_xml))
        result_file.write('Test name: {}\n'.format(self.test_name))
        result_file.write('Features vector location: {}\n'.format(self.feature_vector_path))
        result_file.write('Real images location: {}\n'.format(self.real_images_location))
        result_file.write('Synthetic images location: {}\n'.format(self.synthetic_images_location))
        result_file.write('Features:\n')
        for feature in self.features:
            result_file.write('\tName: {}\n'.format(feature))
            if feature == MATRIX:
                result_file.write('\t\tChannels: {}\n'.format(self.channels))
                result_file.write('\t\tOffsets: {}\n'.format(self.offsets))
        for key, value in self.save_options.items():
            if not value == 'no':
                result_file.write('{} saved in: {}\n'.format(key, value))
        result_file.write('Classifier: {}\n'.format(self.classifier))
        result_file.write('RESULT:\n')
        for key, value in self.decision.items():
            if type(value) == int:
                result_file.write('Prediction set: {}% of training data\tDecision: {}\n'.format(key, value))
            else:
                result_file.write('Prediction set: {}\tDecision: {}\n'.format(key, value))
        result_file.write('##########################\n\n')
        log.result('Result saved in {}'.format(result_path))

    def setup(self):
        #try:
            if self.test_node:
                self.__parse_xml()
            else:
                self.__get_user_input()
            if self.feature_vector == NEW:
                self.images['real'] = self.__load_images(self.real_images_location)
                self.images['synthetic'] = self.__load_images(self.synthetic_images_location)
            self.feature_vector_path = self.feature_vector
            return True
        #except:
         #   return False
