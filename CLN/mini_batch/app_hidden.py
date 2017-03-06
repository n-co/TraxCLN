import numpy
import keras.backend as K


class HiddenSaving():
    """
    A class designed to hold to data of hidden layets in NN.
    :ivar: curr_hidds: a list containing arrays - array per layer. each array is 2d. the size of
                            each array is n_sample*h_dim
    :ivar: n_layers: number of hidden layers for hightway network.
    :ivar: h_dim: amount of nodes in each hidden layer.
    :ivar: n_rel: the number of relations in this network.
    :ivar: n_neight: the number of neigbors each example has i each of the relations. it is always the same number since
                    list was wrapped, and masked.
    :ivar: rel: the relation list, after it has been wrapped to be symetric. the shape of rel is:
                (n_samples, n_rel, n_neigh)
    :ivar: rel_mask: the relation mask, probably indicating which neighbor is true and which is false.
                     the shape of rel_mask is: (n_samples, n_rel, n_neigh)
    """
    def __init__(self, n_samples, n_layers, h_dim, rel, rel_mask):
        """
        :param n_samples: |train|+|valid|+|test|
        :param n_layers: look above.
        :param h_dim: look above.
        :param rel: look above.
        :param rel_mask: look above.
        """
        self.curr_hidds = [numpy.zeros((n_samples, h_dim), dtype='float32') for t in range(n_layers)]
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.n_rel, self.n_neigh = rel.shape[1:]
        self.rel = rel
        self.rel_mask = rel_mask

    def get_context(self, list_ids):
        """
        :param list_ids: nodes for which we would like to get context. can be from any sample type.
        :return: contexts: python list. size of list: n_layers. each on is an array contatining the context of all the
                    requested ids. the context is reshaped to |ids|*r*max_neighbor*hiddem_dim.
                    then mask is used to figure out which is a true neighbor and which is not.
        """
        esp = 1e-8
        n_nodes = len(list_ids)
        mask = self.rel_mask[list_ids]

        contexts, masks = [], []
        for i in range(self.n_layers):
            context = self.curr_hidds[i][self.rel[list_ids].flatten()].reshape\
                ([n_nodes, self.n_rel, self.n_neigh, self.h_dim])
            # context = (n_nodes, n_rel, n_neigh, h_dim)
            # mask = (n_nodes, n_rel, n_neigh) -> sum n_neigh
            context = context * mask[:, :, :, None]
            context = numpy.sum(context, axis=-2) / (numpy.sum(mask, axis=-1) + esp)[:, :, None]
            contexts.append(context)

        return contexts

    def update_hidden(self, list_ids, hidds):
        """
        updates hidden information.
        :param list_ids: ids of elements for which the update should be done.
        :param hidds: the new hidden information.
        :return: nothing.
        """
        for i in range(self.n_layers):
            self.curr_hidds[i][list_ids] = hidds[i]


def get_hidden_funcs(model, n_layers):
    hidd_layer = None
    dense_layer = None
    for layer in model.layers:
        if 'dense' in layer.name and dense_layer is None:
            dense_layer = layer.name

        if 'graph' in layer.name:
            hidd_layer = layer.name

    hidd_funcs = [K.function([model.get_layer('inp_nodes').input, K.learning_phase()],
                             [model.get_layer(dense_layer).output])]

    inps = [model.get_layer('inp_nodes').input]
    for i in range(n_layers - 1):
        inps.append(model.get_layer('inp_context_%d' % i).input)
        get_hidden = K.function(inps + [K.learning_phase()],
                                [model.get_layer(hidd_layer).get_output_at(i)])
        hidd_funcs.append(get_hidden)

    return hidd_funcs


def get_hiddens(model, x, contexts, hidd_funcs):
    hidds = []
    for i in range(len(hidd_funcs)):
        inps = [x] + contexts[:i]
        hidds.append(hidd_funcs[i](inps + [0])[0])

    return hidds
