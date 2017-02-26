from config import *
import gzip
import cPickle
import numpy

def arg_passing(argv):
    i = 1
    #set default args.
    arg_dict = {
        '-data': 'pubmed', #chosing the learning data set: trax, pubmed, movielens, software.
        '-saving': 'pubmed', #log file name.
        '-model': '', #type of NN for each column network. Highway/Dence/CNN?
        '-batch': 100, #batch size for mini-batch version.
        '-y': 1,  # incase of multilabel training, decides on which one net is trained.
        '-nlayers': 10, # number of layers in each highway network.
        '-dim': 50, #number of nodes in each layer of each coloumn network.
        '-shared': 1, #indicator. 1: parameters will be shared between coloumns. 0: no sharing.
        '-nmean': 1, #regulzation factor. shoule be 1<=nmean<=number_of_relations
        '-reg': '', #indicator: dr: dropout. nothing: no dropout.
        '-opt': 'RMS',  # or Adam. an optimizer for paramater tuning.
        '-seed': 1234 #gargabe.
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
    n_nodes = len(rel_list)
    n_rels = len(rel_list[0])
    max_neigh = 0

    for node in rel_list:
        for rel in node:
            max_neigh = max(max_neigh, len(rel))

    rel = numpy.zeros((n_nodes, n_rels, max_neigh), dtype='int64')
    mask = numpy.zeros((n_nodes,n_rels, max_neigh), dtype='float32')

    for i, node in enumerate(rel_list):
        for j, r in enumerate(node):
            n = len(r)
            rel[i, j, : n] = r
            mask[i, j, : n] = 1.0

    return rel, mask

def load_data(path):
    logging.info("load_data - Started.")
    #input: full path to a gzip file, containing cPickle data.
    #output: content of cPickle data.
    f = gzip.open(path, 'rb')
    feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)
    rel_list, rel_mask = create_mask(rel_list)
    logging.info("load_data - Ended.")
    return feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids

class MiniBatchIds():
    # a class designed to generate bacthes of ids (integers) in certain sub-ranges of 0 to n, according to
    # a batch id and total number of batches.
    def __init__(self, n_samples, batch_size=100):
        self.ids = numpy.arange(n_samples)
        self.batch_size = batch_size

    def get_mini_batch_ids(self, batch_id):
        if batch_id == 0:
            numpy.random.shuffle(self.ids)

        return self.ids[self.batch_size * batch_id : self.batch_size * (batch_id + 1)]