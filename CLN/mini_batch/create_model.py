from config import *
import numpy
#from theano import tensor
import tensorflow as tensor
from keras.callbacks import *
from keras import objectives
from keras.layers import *
from keras.models import Model
from sklearn import metrics

from keras.constraints import *
from keras.layers.advanced_activations import *
from graph_layers import *


def create_highway(n_layers, hidden_dim, input_dim, n_rel,
                   n_neigh, n_classes, shared, nmean=1, dropout=True, rel_carry=True):
    # Creates a keras model that can be used, compiled and fitted.
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    trans_bias = - n_layers * bias_factor

    # Creates a Keras Layer. GraphHighway inherits from the Layer Class and overrides it.
    shared_highway = GraphHighway(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                                  init=init, activation=act, transform_bias=trans_bias)

    def highway(is_shared):
        if is_shared == 1:
            return shared_highway
        return GraphHighway(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                            init=init, activation=act, transform_bias=trans_bias)

    # x, rel, rel_mask
    # Input is a method of Keras that creates a tensor. it seems to be a placeholder for now.
    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    # a list of tensors, the size of the number of layers. this is probably because each layer will need to know
    # its context. the shape of each context is the number of relations over the number of hidden dimenstions.
    # seems to be a place holder for now.
    contexts = [Input(shape=(n_rel, hidden_dim), dtype='float32', name='inp_context_%d' % i)
                for i in range(n_layers)]

    # Dence is a a core function of Keras. it inherits from Layer.
    hidd_nodes = Dense(output_dim=hidden_dim, input_dim=input_dim, activation=act)(inp_nodes)
    # wraps hidden nodes with a dropout Keras Class incase of need.
    if dropout:
        hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)

    # this is how we create muliple layers. we iterate and override the hidd_nodes obj and always wrap it.
    # occasionaly we add dropout.
    for i in range(n_layers):
        hidd_nodes = highway(shared)([hidd_nodes, contexts[i]])
        if shared == 0 and i % 5 == 2:
            hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)

    if dropout:
        hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)
    # wraps hidden nodes with a dropout Keras Class incase of need.

    # make the last lauer a dense layer,by wrapping previous hidden layers.
    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)

    # add the top activation that make all probabilities work.
    top_nodes = Activation(activation=top_act)(top_nodes)
    # convert input nodes int a list array, and append contexnt array to this array. creates a model
    # in which input is both input nodes (a tensor of the correct form) and the context array, and output is
    # is top nodes - the hidden layers wrapped with a dense + activation layer.
    model = Model(input=[inp_nodes] + contexts, output=[top_nodes])

    return model


def create_dense(n_layers, hidden_dim, input_dim, n_rel, n_neigh, n_classes, shared, nmean=1, dropout=True):
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    contexts = [Input(shape=(n_rel, hidden_dim), dtype='float32', name='inp_context_%d' % i)
                for i in range(n_layers)]

    hidd_nodes = Dense(input_dim=input_dim, output_dim=hidden_dim, init=init, activation=act)(inp_nodes)
    if dropout:
        hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = GraphDense(input_dim=hidden_dim, output_dim=hidden_dim, init=init,
                                n_rel=n_rel, mean=nmean, activation=act)([hidd_nodes, contexts[i]])
        if dropout:
            hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)

    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)
    model = Model(input=[inp_nodes, contexts], output=[top_nodes])

    return model


def create_highway_no_rel(n_layers, hidden_dim, input_dim, n_classes, shared=1, dropout=True):
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    trans_bias = - n_layers * bias_factor

    shared_highway = Highway(input_dim=hidden_dim, init=init, activation=act, transform_bias=trans_bias)

    def highway(is_shared):
        if is_shared == 1:
            return shared_highway
        return Highway(input_dim=hidden_dim, init=init, activation=act, transform_bias=trans_bias)

    # x, rel, rel_mask
    inp_nodes = Input(shape=(input_dim,), dtype='float32', name='inp_nodes')
    hidd_nodes = Dense(output_dim=hidden_dim, input_dim=input_dim, activation=act)(inp_nodes)
    if dropout:
        hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = highway(shared)(hidd_nodes)

    if dropout:
        hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)
    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)

    model = Model(input=inp_nodes, output=[top_nodes])

    return model


def create_hcnn(n_layers, hidden_dim, input_shape, n_rel, n_neigh, n_classes, shared,
                nmean=1, dropout=True, rel_carry=True):
    logging.info("create_hcnn - Started.")
    logging.debug("create_hcnn parameters: n_layers: %d, hidden_dim: %d, input_shape: %s, n_rel: %d, n_neigh:"
                  " %d, n_classes = %d",
                  n_layers, hidden_dim, str(input_shape), n_rel, n_neigh, n_classes)
    act = 'relu'
    cnn_act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    trans_bias = - n_layers * bias_factor

    image_input_nodes = Input(shape=input_shape, dtype='float32', name='inp_nodes')
    contexts = [Input(shape=(n_rel, hidden_dim), dtype='float32', name='inp_context_%d' % i)
                for i in range(n_layers)]
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
    cnn_nodes = Dense(hidden_dim, activation=cnn_act, name="dnscnn2")(cnn_nodes)

    shared_highway = GraphHighway(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                                  init=init, activation=act, transform_bias=trans_bias)

    def highway(is_shared):
        if is_shared == 1:
            return shared_highway
        return GraphHighway(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                            init=init, activation=act, transform_bias=trans_bias)

    # x, rel, rel_mask
    hidd_nodes = Dense(output_dim=hidden_dim, activation=act)(cnn_nodes)
    if dropout:
        hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = highway(shared)([hidd_nodes, contexts[i]])
        if shared == 0 and i % 5 == 2:
            hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)

    if dropout:
        hidd_nodes = Dropout(dropout_ratio)(hidd_nodes)
    top_nodes = Dense(output_dim=n_classes)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)

    model = Model(input=[image_input_nodes] + contexts, output=[top_nodes])
    logging.info("create_hcnn - Ended.")
    return model
