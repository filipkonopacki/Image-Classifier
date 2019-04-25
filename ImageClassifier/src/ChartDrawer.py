from matplotlib import pyplot as plt
from Utils import IMAGE_TYPES, util, REAL, SYNTHETIC, OFFSETS, CHANNELS
from Log import log

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


vector = read_txt('C:\\private\\Image-Classifier\\ImageClassifier\\feature vectors\\feature_vector_matrices_unq.txt')

real, synth = util.split_features_vector(vector)
data = util.prepare_data(real, synth)

index = data[REAL].index(data[REAL][-1])
unq_real = data[REAL].pop(index)
unq_synth = data[SYNTHETIC].pop(index)

offsets = {}

for image_type in IMAGE_TYPES:
    images = data[image_type]
    im_index = 0
    for offset in OFFSETS:
        offsets[offset, image_type] = []
        for i in range(len(CHANNELS)):
            offsets[offset, image_type].append(images[im_index])
            im_index = im_index + 1
channel_color = ['r', 'g', 'b', 'm', 'c']
image_type_symbol = {}
image_type_symbol[REAL] = '.'
image_type_symbol[SYNTHETIC] = '^'

path = 'C:\\private\\Image-Classifier\\ImageClassifier\\charts\\'
for i in range(len(channel_color)):
    file_name = CHANNELS[i] + '.png'
    for offset in OFFSETS:
        for image_type in IMAGE_TYPES:
            unq = unq_real if image_type == REAL else unq_synth
            type = 'r.' if image_type == REAL else 'g.'
            plt.plot(unq, offsets[offset, image_type][i], type)
    plt.savefig(path + file_name)
    plt.close()





