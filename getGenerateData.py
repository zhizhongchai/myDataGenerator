from PIL import Image
import numpy as np
import os
from keras.utils import np_utils
import threading
import random
from keras.preprocessing.image import img_to_array, array_to_img


def process_line(line):
    tmp = line.strip().split(' ')
    imagefile = tmp[0]
    i = eval(tmp[1])
    j = eval(tmp[2])
    image = Image.open(imagefile)
    data = np.array(image)
    pData = data[i - 75:i + 76, j - 75:j + 76]
    r = eval(tmp[4])

    if (tmp[3] == 'p'):
        imageLabel = 1
    elif (tmp[3] == 'n'):
        imageLabel = 0

    pImage = array_to_img(pData)
    if (r == 1):
        pImage = pImage.rotate(90)

    elif (r == 2):
        pImage = pImage.rotate(180)

    elif (r == 3):
        pImage = pImage.rotate(270)

    elif (r == 5):
        pImage = pImage.transpose(Image.FLIP_LEFT_RIGHT)

    elif (r == 6):
        pImage = pImage.transpose(Image.FLIP_TOP_BOTTOM)

    elif (r == 7):
        pImage = pImage.transpose(Image.FLIP_TOP_BOTTOM)

        pImage = pImage.transpose(Image.FLIP_LEFT_RIGHT)

    pData = np.array(pImage)

    return pData, imageLabel


def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


# process_line('trainImages/9947.png 1\n')
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def generate_arrays(batch_size, path):
    while 1:
        f = open(path)

        list = []
        for line in f:
            list.append(line)
        random.shuffle(list)

        X = []
        Y = []
        cnt = 0
        for count in range(len(list)):
            # print(list[count])
            x, y = process_line(list[count])
            # x = x/255
            x = preprocess_input(x)
            X.append(x)
            Y.append(y)

            cnt += 1
            if cnt == batch_size:
                cnt = 0
                Y = np_utils.to_categorical(Y, 2)
                yield (np.array(X), np.array(Y))
                # print('read')
                X = []
                Y = []
    f.close()

