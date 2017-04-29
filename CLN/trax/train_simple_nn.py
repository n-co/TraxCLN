from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cPickle
import gzip
import numpy as np
import cv2 as ocv
from create_model import *
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot as kplt

np.random.seed(1234)
path = '/vildata/rawdata/Trax/proj_nir_noam/TraxInputData/trax_100_300.pkl.gz'
models_path = "models/"

epochs = 20
batch_size = 128
spe = 16384
nbvs = 16384

print('hello')
# The data, shuffled and split between train and test sets:
f = gzip.open(path, 'rb')
labels, rel_list, train_ids, valid_ids, test_ids, paths, batches = cPickle.load(f)

p_train = paths[0:16384]
y_train = labels[0:16384]
amount_of_train_samples = len(p_train)

p_test = paths[16384:16384+2048]
y_test = labels[16384:16384+2048]
amount_of_test_samples = len(p_test)

num_classes = np.max(labels) + 1

print('p_train shape:', p_train.shape)
print(p_train.shape[0], 'train samples')
print(p_test.shape[0], 'test samples')
print('y_train shape ', y_train.shape)
print('y_test shape', y_test.shape)
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('y_train shape ', y_train.shape)
print('y_test shape', y_test.shape)
print('number of classes: ', num_classes)
print(len(p_train))
print(len(p_test))


def read_from_disk(path):
    res = ocv.imread(path)
    res = res.astype('float32')
    res /= 255
    # ocv.imshow("path",res)
    # ocv.waitKey(0)
    return res


def generator_creator(paths, lables):
    while True:
        i = 0
        while i < len(paths):
            x_list = [read_from_disk(p) for p in paths[i:i + batch_size]]
            # x_list = [np.expand_dims(x, axis=0) for x in x_list]
            y_list = lables[i:i + batch_size]
            # y_list = [np.expand_dims(y, axis=0) for y in y_list]
            i += batch_size
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            yield x_list, y_list


train_gen = generator_creator(p_train, y_train)
test_gen = generator_creator(p_test, y_test)
example_x = read_from_disk(paths[0])
print('example x shape:', example_x.shape)

model = create_cnn(example_x.shape, num_classes)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.summary()
kplt(model, to_file=models_path + 'simple_nn' + '.png', show_shapes=True)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(generator=train_gen, samples_per_epoch=spe, nb_epoch=epochs,
                    verbose=1, callbacks=None,
                    validation_data=test_gen, nb_val_samples=nbvs,
                    class_weight=None, max_q_size=1, nb_worker=1,
                    pickle_safe=False, initial_epoch=0, )
