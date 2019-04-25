from matplotlib import pyplot as plt
from Utils import IMAGE_TYPES

def read_txt(path):
    vector = []
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
        vector.append(tmp_vector)
    return vector


vector = read_txt('C:\\Users\\Tazik\\Desktop\\praca inżynierska\\płyta\\feature vectors\\feature_vector_matrices_unq.txt')
unq = []


