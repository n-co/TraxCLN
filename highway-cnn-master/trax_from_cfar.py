'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

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

from keras.utils.np_utils import to_categorical

path = '/vildata/rawdata/Trax/proj_nir_noam/TraxInputData/trax_200_600.pkl.gz'
models_path = "models/"

batch_size = 32
num_classes = 10
epochs = 10
data_augmentation = False

print('hello')
# The data, shuffled and split between train and test sets:
f = gzip.open(path, 'rb')
labels, rel_list, train_ids, valid_ids, test_ids, paths, batches = cPickle.load(f)

p_train = paths[0:128]
y_train = labels[0:128]
amount_of_train_samples = len(p_train)

p_test = paths[0:128]
y_test = labels[0:128]
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
    return res


# def train_gen_creator():
#     while True:
#         for i in range(len(p_train)):
#             x = read_from_disk(p_train[i])
#             x = np.expand_dims(x, axis=0)
#             y = y_train[i]
#             y = np.expand_dims(y, axis=0)
#             yield x, y
#
#
# def test_gen_creator():
#     while True:
#         for i in range(len(p_test)):
#             x = read_from_disk(p_test[i])
#             x = np.expand_dims(x, axis=0)
#             y = y_test[i]
#             y = np.expand_dims(y, axis=0)
#             yield x, y

def train_gen_creator():
    while True:
        i = 0
        while i < len(p_train):
            x_list = [read_from_disk(p) for p in p_train[i:i+batch_size]]
            # x_list = [np.expand_dims(x, axis=0) for x in x_list]
            y_list = y_train[i:i+batch_size]
            # y_list = [np.expand_dims(y, axis=0) for y in y_list]
            i += batch_size
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            yield x_list, y_list


def test_gen_creator():
    while True:
        i = 0
        while i < len(p_train):
            x_list = [read_from_disk(p) for p in p_test[i:i+batch_size]]
            # x_list = [np.expand_dims(x, axis=0) for x in x_list]
            y_list = y_test[i:i+batch_size]
            # y_list = [np.expand_dims(y, axis=0) for y in y_list]
            i += batch_size
            yield np.array(x_list), np.array(y_list)

train_gen = train_gen_creator()
test_gen = test_gen_creator()
example_x = read_from_disk(paths[0])
print('example x shape:', example_x.shape)


model = Sequential()
# TODO: try put our layers
model.add(Conv2D(4, 3, 3, border_mode='same', input_shape=example_x.shape))
model.add(Activation('relu'))
model.add(Conv2D(4, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(8, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.summary()

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# model.fit_generator(train_gen, samples_per_epoch=200, nb_epoch=10,
#                     verbose=1, callbacks=None,
#                     validation_data=test_gen, nb_val_samples=100,
#                     class_weight=None, max_q_size=1, nb_worker=1,
#                     pickle_safe=False, initial_epoch=0,)

model.fit_generator(train_gen,
                    amount_of_train_samples // batch_size,
                    epochs,
                    1,
                    None,
                    test_gen,
                    2,
                    None,
                    10,
                    1,
                    False,
                    0,
                    )
