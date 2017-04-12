'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

import numpy as np
import cv2 as ocv
import sys
import time
import gzip
import cPickle
import logging
from keras.optimizers import *
from keras.objectives import *
import keras
from  keras.constraints import *
from keras.utils import plot_model as kplt
from keras.layers import *
from keras.models import Sequential
from keras.models import Model
sys.path.insert(0, '../../')
import logger
from tools import *

logging.getLogger().setLevel(logging.DEBUG)
run_mode = 'run'
product_height = 200
product_width = 600
product_channels = 3
dropout_ration_cnn = 0.2
batch_size = 256
n_epochs = 200
init = 'glorot_normal'
act = 'relu'
cnn_act = 'relu'
top_act = 'softmax'
path = '/vildata/rawdata/Trax/proj_nir_noam/TraxInputData/trax.pkl.gz'
models_path = "models/"


class MiniBatchIds:
    """
    A class designed to generate bacthes of ids (integers) in certain sub-ranges of 0 to some n.
    :ivar ids: an array of all integers in the interval [0,n_samples-1] (including edges)
    :ivar: batch_size: the size of a single batch.
    """
    def __init__(self, n_samples, batch_size):
        """
        :param n_samples: the number of samples in the entire training set.
        :param batch_size: the size of a single batch.
        """
        self.ids = np.arange(n_samples)
        self.batch_size = batch_size

    def get_mini_batch_ids(self, batch_id):
        """
        :param: batch_id: the index of the current batch. relevant ids can be calculated from the list by this index.
        :return: an array of indexes, continious, decribing ids of the burrent batch.
        :operations: when batch_id is 0, the ids array is suffled. this happens at the begining of every epoch, so every
                     epoch covers all train ids, but in a different order.
        """
        if batch_id == 0:
            np.random.shuffle(self.ids)
        return self.ids[self.batch_size * batch_id: self.batch_size * (batch_id + 1)]


def load_data(path):
    """
    loads data from a pcl file into memory.
    :param: path: full path to a gzip file, containing cPickle data.
    :return: content of cPickle data, in seperate arrays all of the size.
            this means, for every arr returned, a.shape[0] is the same
    """
    logging.info("load_data - Started.")
    f = gzip.open(path, 'rb')
    labels, rel_list, train_ids, valid_ids, test_ids, paths,batches = cPickle.load(f)
    logging.debug(str(paths))
    labels = labels.astype('int64')
    logging.info("load_data - Ended.")
    return labels, train_ids, valid_ids, test_ids, paths


def extract_featurs(feats_paths, ids):
    """
    :param feats_paths: paths to all products.
    :param ids: requested ids.
    :return: a tensor containing the feautures in desired format.
    """
    logging.info("extract_featurs: Started.")
    logging.info("there are %d examples. batch size is %d." % (len(feats_paths), len(ids)))
    feats = np.zeros((len(ids), product_width, product_height, product_channels), dtype=type(np.ndarray))

    for ii in range(len(ids)):
        res = ocv.imread(feats_paths[ids[ii]])
        res = res.astype('float32')
        res /= 255
        feats[ii] = res

    ans = feats
    logging.info("extract_featurs: Ended.")
    return ans

def create_cfer2_model():
    model = Sequential()

    model.add(Conv2D(4, (3, 3), padding='same',
                     input_shape=valid_x.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(4, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def create_cnn_model():
    image_input_nodes = Input(shape=example_x[0].shape, dtype='float32', name='inp_nodes')
    cnn_nodes = image_input_nodes
    #TODO: verify the order of args height and width
    cnn_nodes = Convolution2D(4, 3, 3, input_shape=(product_channels, product_height, product_width), activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Convolution2D(4, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    cnn_nodes = Convolution2D(8, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Convolution2D(8, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    cnn_nodes = Convolution2D(16, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Convolution2D(16, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    cnn_nodes = Convolution2D(32, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Convolution2D(32, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    cnn_nodes = Convolution2D(64, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Convolution2D(64, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    cnn_nodes = Convolution2D(128, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Convolution2D(128, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    cnn_nodes = Flatten()(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Dense(1024, activation='relu', W_constraint=maxnorm(3), name="dnscnn1")(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    top_nodes = Dense(output_dim=n_classes)(cnn_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)

    model = Model(input=image_input_nodes, output=[top_nodes])

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    kplt(model, to_file=models_path + 'trax_cnn' + '.png', show_shapes=True)
    return model

labels, train_ids, valid_ids, test_ids, paths = load_data(path)
n_classes = np.max(labels)
if n_classes > 1:
    n_classes += 1
n_classes = abs(n_classes)
train_sample_size = len(train_ids)
valid_sample_size = len(valid_ids)
test_sample_size = len(test_ids)
example_x = extract_featurs(paths, [0])
valid_x = extract_featurs(paths, valid_ids)
valid_y = labels[valid_ids]
test_x = extract_featurs(paths, test_ids)
test_y = labels[test_ids]
n_batchs = train_sample_size // batch_size
if train_sample_size % batch_size > 0:
    n_batchs += 1
model = create_cfer2_model()
train_mini_batch_ids_generator = MiniBatchIds(train_sample_size, batch_size=batch_size)



test_y = keras.utils.to_categorical(test_y, n_classes)
valid_y = keras.utils.to_categorical(valid_y, n_classes)


for e in range(n_epochs):
    for b in range(n_batchs):
        batch_start = time.time()
        mini_batch_ids = train_ids[train_mini_batch_ids_generator.get_mini_batch_ids(b)]
        train_x = extract_featurs(paths, mini_batch_ids)
        train_y = labels[mini_batch_ids]
        train_y = keras.utils.to_categorical(train_y, n_classes)
        logging.error("shape of train_y: %s" % str(train_y.shape))
        valid_data = [valid_x, valid_y]
        # model.fit(train_x, numpy.expand_dims(train_y, -1), validation_data=valid_data, verbose=0,
        #           nb_epoch=1, batch_size=train_x.shape[0], shuffle=False)
        model.fit([train_x] + [], train_y, batch_size= train_x.shape[0], nb_epoch=1, validation_data=valid_data)
        batch_end = time.time()
        logging.info("batch %d in epoch %d. is done. time: %f." % (b, e, batch_end - batch_start))




