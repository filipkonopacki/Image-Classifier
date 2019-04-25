"""
# Class responsible for displaying and saving logs
"""


# Libraries
import os
import datetime


# Log constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INFO = 'INFO'
ERROR = 'ERROR'
INPUT = 'INPUT'
WARN = 'WARNING'
RESULT = 'RESULT'
REQUEST = 'REQUEST'


class Log:

    def __init__(self):
        self.log_file_path = None
        self.logs_dir = None
        self.log_file = None

    def setup(self):
        self.logs_dir = ROOT_DIR + '\\logs'
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
            print('Local directory for saving logs created at: ' + self.logs_dir)

        today = datetime.date.today()
        self.log_file_path = self.logs_dir + '\\log__' + today.__str__() + '.txt'
        self.log_file = open(self.log_file_path, 'a')
        self.info('********** New session started *********')

    def __create_log(self, type, message):
        time = datetime.datetime.now()
        time = time.strftime('%X')
        log = '[' + type + ' ' + time + ']\t' + message
        print(log)
        self.log_file.write(log + '\n')

    def info(self, message):
        self.__create_log(INFO, '\t' + message)

    def error(self, message):
        self.__create_log(ERROR, message)

    def input(self, message):
        self.__create_log(INPUT, message)

    def result(self, message):
        self.__create_log(RESULT, message)

    def request(self, message):
        self.__create_log(REQUEST, message)

    def warning(self, message):
        self.__create_log(WARN, message)

log = Log()
log.setup()