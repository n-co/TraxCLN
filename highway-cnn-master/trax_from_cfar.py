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

np.random.seed(1234)
path = '/vildata/rawdata/Trax/proj_nir_noam/TraxInputData/trax_100_300.pkl.gz'
models_path = "models/"

epochs = 100
batch_size = 128
spe =128
nbvs =128

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
    # ocv.imshow("path",res)
    # ocv.waitKey(0)
    return res


def generator_creator(paths, lables):
    global count
    while True:
        i = 0
        while i < len(paths):
            x_list = [read_from_disk(p) for p in paths[i:i + batch_size]]
            y_list = lables[i:i+batch_size]
            i += batch_size
            yield np.array(x_list), np.array(y_list)


train_gen = generator_creator(p_train,y_train)
test_gen = generator_creator(p_test,y_test)
example_x = read_from_disk(paths[0])
print('example x shape:', example_x.shape)


model = Sequential()
# TODO: try put our layers
model.add(Conv2D(16, 3, 3, border_mode='same', input_shape=example_x.shape))
model.add(Activation('relu'))
model.add(Conv2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
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

# it is better when samples_per_epoch % amount_of_train_samples == 0
model.fit_generator(generator= train_gen, samples_per_epoch=spe, nb_epoch=epochs,
                    verbose=1, callbacks=None,
                    validation_data=test_gen, nb_val_samples=nbvs,
                    class_weight=None, max_q_size=1, nb_worker=1,
                    pickle_safe=False, initial_epoch=0,)

# model.fit_generator(train_gen,
#                     amount_of_train_samples // batch_size,
#                     epochs,
#                     1,
#                     None,
#                     test_gen,
#                     2,
#                     None,
#                     10,
#                     1,
#                     False,
#                     0,
#                     )
