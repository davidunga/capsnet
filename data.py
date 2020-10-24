import numpy as np
import os
import csv
import torch as T

# ------------------------------------------------------------------------------

DATA_DIR = '' # path to folder that contains mnist_train.csv & mnist_test.csv

# ------------------------------------------------------------------------------

def load(train_or_test, num_samples=None, num_classes=10):
    """ Load MNIST data from csv.
    :param train_or_test: 'train' / 'test'
    :param num_samples: number of samples to get (None = all)
    :param num_classes: number of classes to use (None = all)
    :return:
        imgs - (#samples, 28, 28, 1) float (0-1) tensor
        labels - (#samples, #classes) float one hot
    """
    fname = DATA_DIR + os.path.sep + 'mnist_' + train_or_test + '.csv'
    assert os.path.isfile(fname), "CSV not found in data dir"

    imgs, labels = [], []
    with open(fname) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            label = int(row[0])
            if label >= num_classes:
                continue
            img = np.array([int(x) for x in row[1:]]).reshape((28, 28, 1))
            imgs.append(img)
            labels.append(label)
            if num_samples is not None and len(imgs) == num_samples:
                break

    imgs = T.tensor(imgs, dtype=T.float32) / 255
    labels = T.nn.functional.one_hot(T.tensor(labels)).to(T.float32)

    return imgs, labels


def _show(imgs, labels):
    import matplotlib.pyplot as plt
    for i, img in enumerate(imgs):
        plt.imshow(img[...,0], cmap='gray')
        plt.title('label=' + str(int(T.argmax(labels[i]))))
        plt.draw()
        try:
            plt.waitforbuttonpress()
        except:
            break


if __name__ == "__main__":
    # test
    imgs, labels = load('train', num_samples=50, num_classes=10)
    _show(imgs, labels)


