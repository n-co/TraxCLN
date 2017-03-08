from config import *
import gzip
import cPickle
import numpy


def arg_passing(argv):
    """
    a function to process input arguments.
    :param argv: the command line args that were passed.
    :return: arg_dict: a json dictionary containing args in the desire.
    """
    i = 1
    # set default args.
    arg_dict = {
        '-data': 'pubmed',  # chosing the learning data set: trax, pubmed, movielens, software.
        '-saving': 'pubmed',  # log file name.
        '-model': '',  # type of NN for each column network. Highway/Dence/CNN?
        '-batch': 100,  # batch size for mini-batch version.
        '-y': 1,  # incase of multilabel training, decides on which one net is trained.
        '-nlayers': 10,  # number of layers in each highway network.
        '-dim': 50,  # number of nodes in each layer of each coloumn network.
        '-shared': 1,  # indicator. 1: parameters will be shared between coloumns. 0: no sharing.
        '-nmean': 1,  # regulzation factor. shoule be 1<=nmean<=number_of_relations
        '-reg': '',  # indicator: dr: dropout. nothing: no dropout.
        '-opt': 'RMS',  # or Adam. an optimizer for paramater tuning.
        '-seed': 1234  # used to make random decisions repeat.
    }
    # Update args to contain the user's desired configuration.
    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i+1]
        i += 2
    # Update data types for arguments.
    arg_dict['-nlayers'] = int(arg_dict['-nlayers'])
    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-shared'] = int(arg_dict['-shared'])
    arg_dict['-nmean'] = int(arg_dict['-nmean'])
    arg_dict['-seed'] = int(arg_dict['-seed'])
    arg_dict['-y'] = int(arg_dict['-y'])
    return arg_dict


def create_mask(rel_list):
    """
    reformats a given relation list to contain list elements all of the same size.
    :param rel_list: a list of size n_nodes = |sample|+|valid|+|test|. each elemt is a list of size n_rels = r.
                  each of these lists is of a different length
    :return: rel: same as relation list, but padded with 0's were no relation applies.
              TODO: this might mean that all are in a relation with sample 0. check.
    :return: mask: formmated exactly like rel, but does not contain actual ids for samples in realtion,
                    but rather 0 or 1.
              TODO: is this used to handle the concern above?
    """
    n_nodes = len(rel_list)
    n_rels = len(rel_list[0])
    max_neigh = 0

    for sample in rel_list:
        for rel in sample:
            max_neigh = max(max_neigh, len(rel))

    rel = numpy.zeros((n_nodes, n_rels, max_neigh), dtype='int64')
    mask = numpy.zeros((n_nodes, n_rels, max_neigh), dtype='float32')

    for i, sample in enumerate(rel_list):  # go over all samples, while saving a reference to to
                                            # the index of a sample and the sample itself.
        for j, r in enumerate(sample):  # go over all relations of an example, while saving
                                        # a reference to the index of the relation and the relation itself.
            n = len(r)
            rel[i, j, : n] = r
            mask[i, j, : n] = 1.0

    return rel, mask


def load_data_trax(path):
    """
    loads data from a pcl file into memory.
    :param: path: full path to a gzip file, containing cPickle data.
    :return: content of cPickle data, in seperate arrays all of the size.
            this means, for every arr returned, a.shape[0] is the same
    """
    logging.info("load_data - Started.")
    f = gzip.open(path, 'rb')
    feats, labels, rel_list, train_ids, valid_ids, test_ids,paths = cPickle.load(f)
    rel_list, rel_mask = create_mask(rel_list)
    logging.info("load_data - Ended.")
    return feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids,paths


def load_data(path):
    """
    loads data from a pcl file into memory. for every dataset but trax.
    :param: path: full path to a gzip file, containing cPickle data.
    :return: content of cPickle data, in seperate arrays all of the size.
            this means, for every arr returned, a.shape[0] is the same
    """
    logging.info("load_data - Started.")
    f = gzip.open(path, 'rb')
    feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)
    rel_list, rel_mask = create_mask(rel_list)
    logging.info("load_data - Ended.")
    return feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids

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
        self.ids = numpy.arange(n_samples)
        self.batch_size = batch_size

    def get_mini_batch_ids(self, batch_id):
        """
        :param: batch_id: the index of the current batch. relevant ids can be calculated from the list by this index.
        :return: an array of indexes, continious, decribing ids of the burrent batch.
        :operations: when batch_id is 0, the ids array is suffled. this happens at the begining of every epoch, so every
                     epoch covers all train ids, but in a different order.
        """
        if batch_id == 0:
            numpy.random.shuffle(self.ids)
        return self.ids[self.batch_size * batch_id: self.batch_size * (batch_id + 1)]


def extract_featurs(feats_paths, ids, task):
    """
    :param feats_paths: paths to all products.
    :param ids: requested ids.
    :param: task: the task this NN is performing
    :return: a tensor containing the feautures in desired format.
    """
    logging.debug("loading images from disk - started.")
    logging.debug("db size is %d batch size is %d task is %s" % (len(feats_paths),len(ids), task))
    size_of_db = len(feats_paths)
    feats = np.zeros((size_of_db, product_height, product_width, product_channels), dtype=type(np.ndarray))
    ans = None
    if task == 'trax':
        for iden in ids:
            feats[iden] = ocv.imread(feats_paths[iden])
        ans = feats[ids]
    else:
        ans = feats_paths[ids]
    logging.debug("loading images from disk - ended.")
    return ans
