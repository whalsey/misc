"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

from __future__ import print_function

#### Libraries

# Standard library
import cPickle
import gzip
import os.path
import random

# Third-party libraries
import numpy as np
import scipy.ndimage.interpolation

import matplotlib.pyplot as plt

def sign(a):
    return -1 if a < 0 else 1

print("Expanding the MNIST training set")

if os.path.exists("../data/mnist_expanded.pkl.gz"):
    print("The expanded training set already exists.  Exiting.")
else:
    f = gzip.open("../data/mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    expanded_training_pairs = []
    j = 0 # counter

    # for each image in the training data
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))

        j += 1
        if j % 1000 == 0: print("Expanding image number ", j)

        # create four new images with shifts and rotations

        for _ in range(4):
            # calculate x shift
            shift_x = random.randint(-3, 3)

            # calculate y shift
            shift_y = random.randint(-3, 3)

            new_img = np.roll(image, shift_x, 0)
            new_img = np.roll(new_img, shift_y, 1)

            # pad the shifted area with 0's
            # todo - will add this later *(though it does not seem necessary)
            # if sign(shift_x) == 1:
            #     new_img[:shift_x][:] = np.zeros((shift_x, 28))
            # else:
            #     new_img[28-shift_x:][:] = np.zeros((shift_x, 28))
            #
            # if sign(shift_y) == 1:
            #     new_img[:][:shift_y] = np.zeros((28, shift_y))
            # else:
            #     new_img[:][28-shift_y:] = np.zeros((28, shift_y))

            # calculate degree of rotation
            degree = (random.random() - 0.5) * 90

            new_img = scipy.ndimage.interpolation.rotate(new_img, degree, reshape=False)

            # plt.imshow(new_img)
            #
            # plt.pause(0.01)
            # plt.clf()

            expanded_training_pairs.append((np.reshape(new_img, 784), y))

    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]

    print("Saving expanded data. This may take a few minutes.")

    f = gzip.open("../data/mnist_expanded.pkl.gz", "w")
    cPickle.dump((expanded_training_data, validation_data, test_data), f)
    f.close()
