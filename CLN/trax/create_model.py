from config import *
from graph_layers import *

from keras.layers import *
from keras.models import Model,Sequential
from keras.constraints import *


def flat_images(flat_method, input_image_nodes, hidden_dim, pooling):
    """

    :param flat_method: using flatting or convolution
    :param input_image_nodes: a layer to be flattened.
    :param hidden_dim: passed along.
    :param pooling: passed along.
    :return:
    """
    cnn_nodes = input_image_nodes
    if flat_method == 'c':
        cnn_nodes = flat_by_conv(input_image_nodes, hidden_dim,pooling)
    elif flat_method == 'f':
        cnn_nodes = flat_by_flat(input_image_nodes)
    return cnn_nodes


def flat_by_conv(image_input_nodes, hidden_dim,pooling):
    """
    flates an input layer using a sequence of convolutions and pooling layers.
    :param image_input_nodes: a layer to be flattened.
    :param hidden_dim: target size for input to highway part.
    :param pooling: boolean. use MaxAveragePooling in the end, or standart Dense layers.
    :return:
    """
    cnn_act = 'relu'
    nb_filer = 16
    number_of_conv_structs = 3
    cnn_nodes = image_input_nodes

    cnn_nodes = Convolution2D(nb_filer, 3, 3, input_shape=(product_channels, product_height, product_width),
                              activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    cnn_nodes = Convolution2D(nb_filer, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
    cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    for i in range(number_of_conv_structs):
        nb_filer *= 2
        cnn_nodes = Convolution2D(nb_filer, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
        cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
        cnn_nodes = Convolution2D(nb_filer, 3, 3, activation=cnn_act, border_mode='same')(cnn_nodes)
        cnn_nodes = MaxPooling2D(pool_size=(2, 2))(cnn_nodes)

    if pooling:
        cnn_nodes = GlobalAveragePooling2D()(cnn_nodes)
        cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
    else:
        cnn_nodes = Flatten()(cnn_nodes)
        cnn_nodes = Dense(2048, activation='relu', W_constraint=maxnorm(3), name="dnscnn1")(cnn_nodes)
        cnn_nodes = Dropout(dropout_ration_cnn)(cnn_nodes)
        cnn_nodes = Dense(hidden_dim, activation=cnn_act, name="dnscnn2")(cnn_nodes)

    return cnn_nodes


def flat_by_flat(image_input_nodes):
    """
    flattens input layer using simple flat.
    :param image_input_nodes:
    :return:
    """
    cnn_nodes = Flatten()(image_input_nodes)
    return cnn_nodes


def create_cnn(input_shape,n_classes,pooling=True):
    """
    creates a Keras model of a standart CNN.
    :param input_shape: shape of input.
    :param n_classes: number of classes in model.
    :param pooling: use Global Average Pooling or not.
    :return: a keras model.
    """
    logging.debug("create_cnn: started.")
    input_image_nodes = Input(shape=input_shape, dtype='float32', name='inp_nodes')
    cnn_nodes = flat_images('c', input_image_nodes, 2048, pooling)
    top_nodes = Dense(output_dim=n_classes)(cnn_nodes)
    top_nodes = Activation(activation='softmax')(top_nodes)
    model = Model(input=[input_image_nodes ], output=[top_nodes])
    logging.debug("create_cnn: ended.")
    return model


def create_hcnn_relation(n_layers, hidden_dim, input_shape, n_rel, n_neigh, n_classes, shared, nmean=1, dropout=True,
                         rel_carry=True, flat_method='c',pooling=True):
    """
    create a keras model of a CNN using relations implemented with column highway layers.
    :param n_layers: internal number of layers in column.
    :param hidden_dim: internal width of column.
    :param input_shape: shape of input.
    :param n_rel: number of relations to test.
    :param n_neigh: maxmimal number of examples in each relation.
    :param n_classes: number of classes in model.
    :param shared: share paramters between layers of columns, or not.
    :param nmean: regulazation for internal layers.
    :param dropout: use dropout or not.
    :param rel_carry: ?
    :param flat_method: flat images using convolutions or simple flattening.
    :param pooling: use Global Average Pooling or not.
    :return: a keras model.
    """
    logging.debug("create_hcnn_relation: started.")
    act = 'relu'
    top_act = 'softmax' if n_classes > 1 else 'sigmoid'
    n_classes = abs(n_classes)
    init = 'glorot_normal'

    trans_bias = - n_layers * 0.1

    shared_highway = GraphHighwayByRel(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                                       init=init, activation=act, transform_bias=trans_bias)

    def highway(shared):
        if shared == 1:
            return shared_highway
        return GraphHighwayByRel(input_dim=hidden_dim, n_rel=n_rel, mean=nmean, rel_carry=rel_carry,
                                 init=init, activation=act, transform_bias=trans_bias)

    # x, rel, rel_mask
    input_image_nodes = Input(shape=input_shape, dtype='float32', name='inp_nodes')
    inp_rel = Input(shape=(n_rel, n_neigh), dtype='int32', name='inp_rel')
    inp_rel_mask = Input(shape=(2, n_rel, n_neigh), dtype='float32', name='inp_rel_mask')

    cnn_nodes = flat_images(flat_method, input_image_nodes, hidden_dim, pooling)

    hidd_nodes = Dense(output_dim=hidden_dim, input_dim=input_shape, activation=act)(cnn_nodes)
    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)

    for i in range(n_layers):
        hidd_nodes = highway(shared)([hidd_nodes, inp_rel, inp_rel_mask])
        if shared == 0 and i % 5 == 2:
            hidd_nodes = Dropout(0.5)(hidd_nodes)

    if dropout: hidd_nodes = Dropout(0.5)(hidd_nodes)
    top_nodes = Dense(output_dim=n_classes, input_dim=hidden_dim)(hidd_nodes)
    top_nodes = Activation(activation=top_act)(top_nodes)

    model = Model(input=[input_image_nodes, inp_rel, inp_rel_mask], output=[top_nodes])
    logging.debug("create_hcnn_relation: ended.")
    return model
