"""
# Main class, manages whole process of image classification
"""


import sys
from time import sleep
from xml.dom import minidom


from Log import log
from Test import Test
from FeaturesManager import FeaturesManager
from Classifiers import Classifiers


FAIL = 'FAIL'
PASS = 'PASS'


class ImageClassifier:

    def __init__(self):
        self.test_xml = None
        self.test_xml_path = None
        self.test_suite = []

    def setup(self):
        if len(sys.argv) > 1:
            if sys.argv[1]:
                self.test_xml_path = sys.argv[1]
                try:
                    self.test_xml = minidom.parse(self.test_xml_path)
                except:
                    log.error('Error during parsing xml file.')
                    raise NameError('Invalid input: {}'.format(self.test_xml))
                tests = self.test_xml.getElementsByTagName('tests')[0]
                for test in tests.childNodes:
                    try:
                        test_name = test.attributes['name'].value
                        test_obj = Test(test, test_name)
                        self.test_suite.append(test_obj)
                    except:
                        pass
        else:
            test_obj = Test()
            self.test_suite.append(test_obj)

    def run(self):
        self.setup()
        if len(self.test_suite) == 0:
            log.error('Test suite is empty. Error during tests creation.')
            raise NameError('Test creation error')
        for test in self.test_suite:
            if not test.setup():
                log.error('Error during setting up test suite for test {}.'.format(test.test_name))
                raise NameError('Invalid input')
            log.info('**********Starting test {} **********'.format(test.test_name))
            features_manager = FeaturesManager(test)
            features_manager.get_features_vector()
            for prediction in test.prediction_set:
                try:
                    prediction = int(prediction)
                except:
                    features_manager.get_features_vector(prediction)
            classifiers = Classifiers(test)
            classifiers.run()
            test.save_result(self.test_xml_path)
            log.info('********** Test {} passed **********'.format(test.test_name))
            test = None
            classifiers = None


if __name__ == "__main__":
    result = FAIL
    try:
        classifier = ImageClassifier()
        classifier.run()
        result = PASS
    except NameError:
        sleep(10)
        raise
    finally:
        log.result('Program stops with status: ' + result)
