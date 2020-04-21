import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def gen_images():
    data = unpickle('images/data_batch_1')
    images = data[b'data']

    for ind, im in enumerate(images):
        im = [[im[i],im[i+1024],im[i+2048]] for i in range(1024)]
        im = Image.fromarray(np.reshape(im,(32,32,3))).convert('RGB')
        im.save("images/" + str(ind) + ".png")