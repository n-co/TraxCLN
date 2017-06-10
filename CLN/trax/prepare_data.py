from config import *
import numpy as np
import cv2 as ocv
import gzip
import cPickle
import random
from keras.optimizers import *
from keras.utils.np_utils import to_categorical


def process_input_args(argv):
    """
    a function to process input arguments.
    :param argv: the command line args that were passed.
    :return: arg_dict: a json dictionary containing args in the desire.
    """
    logging.debug("process_input_args: Started.")
    # set default args.
    arg_dict = {
        '-dataset': 'trax_100_300_filtered',  # chosing the learning data set.
        '-model_type': 'HCNN',  # type of NN for each column network. HCNN/CNN
        '-batch_type': 'constant',  # constant/random/probe
        '-batch_size': '5',  # batch size for mini-batch version.
        '-constant_batch_size': '128',  # when selecting a constant batch size, this is the one.
        '-nlayers': '10',  # number of layers in each highway network.
        '-dim': '400',  # number of nodes in each layer of each coloumn network.
        '-shared': '1',  # indicator. 1: parameters will be shared between coloumns. 0: no sharing.
        '-nmean': '2',  # regulzation factor. shoule be 1<=nmean<=number_of_relations
        '-dropout': '1',  # 1: use dropout. 0- dont.
        '-flatmethod': 'c',  # 'c' -CNN. 'f' - Flat. else - No Flat.
        '-pooling': '0',  # 1 - use max pooling. 0 - dont use max pooling.
        '-seed': '1234',  # used to make random decisions repeat.
        '-opt': 'RMS2',  # or Adam. an optimizer for paramater tuning.
        '-learning_rate': 0.0001, # learning rate for optimizer.
        '-number_of_epochs': 100, #number of epochs to train model.
        '-notes': None

    }
    # Update args to contain the user's desired configuration.
    i = 1
    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i + 1]
        i += 2


    arg_dict['-saving'] = arg_dict['-dataset'] + '_' + arg_dict['-model_type'] + '_' + arg_dict['-flatmethod'] + '_' + arg_dict['-pooling']
    arg_dict['-dataset']=data_sets_dir + arg_dict['-dataset'] + '.pkl.gz'
    # model type
    # batch type
    arg_dict['-batch_size'] = int(arg_dict['-batch_size'])
    arg_dict['-constant_batch_size'] = int(arg_dict['-constant_batch_size'])
    arg_dict['-nlayers'] = int(arg_dict['-nlayers'])
    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-shared'] = int(arg_dict['-shared'])
    arg_dict['-nmean'] = int(arg_dict['-nmean'])
    arg_dict['-dropout'] = True if arg_dict['-dropout']=='1' else False
    arg_dict['-pooling'] = True if arg_dict['-pooling'] == '1' else False
    # flatmethod
    arg_dict['-seed'] = int(arg_dict['-seed'])
    #opt
    arg_dict['-learning_rate'] = float(arg_dict['-learning_rate'])
    arg_dict['-number_of_epochs'] = int(arg_dict['-number_of_epochs'])
    logging.debug("process_input_args: Ended.")
    return arg_dict


def get_global_configuration(argv):
    """

    :param argv: the command line global_config that were passed.
    :return: many variables to be used as globals in main.
    """
    logging.debug("get_global_configuration - Started.")

    global_config = process_input_args(argv)
    np.random.seed(global_config['-seed'])

    labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, paths, batches = load_data(global_config['-dataset'])

    global_config['-n_classes'] = np.max(labels) + 1
    global_config['-train_size'] = len(train_ids)
    global_config['-valid_size'] = len(valid_ids)
    global_config['-test_size'] = len(test_ids)
    global_config['-image_padding'] = 'zero_padding'
    labels = to_categorical(labels, global_config['-n_classes'])

    logging.debug("get_global_configuration - Ended.")
    return global_config, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids,paths, batches


def load_data(path):
    """
    loads data from a pcl file into memory.
    :param: path: full path to a gzip file, containing cPickle data.
    :return: content of cPickle data, in seperate arrays all of the size.
            this means, for every arr returned, a.shape[0] is the same
    """
    logging.debug("load_data - Started.")
    f = gzip.open(path, 'rb')
    labels, rel_list, train_ids, valid_ids, test_ids, paths, batches = cPickle.load(f)
    logging.debug(str(paths))
    rel_list, rel_mask = create_mask_relation(rel_list)
    labels = labels.astype('int64')
    logging.debug("load_data - Ended.")
    return labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, paths, batches


class MiniBatchIdsByProbeId:
    def __init__(self, probe_serials, n_samples, number_of_probes, probes_per_batch):
        '''
        designed to handle a single Sample!(train/valid/test).
        :param probe_serials: a list of length 'n_samples'. the nth element in the list is a serial of the probe that
         contains the nth product (bottle) in the csv file.
        :param n_samples: number of products
        :param number_of_probes:
        :param probes_per_batch:
        '''
        logging.debug("MiniBatchIdsByProbeId constructor")
        logging.debug("can handle: probes: %d. products: %d. probes in batch: %d." % (
            n_samples, number_of_probes, probes_per_batch))
        # probes_arr = a numpy array of lists. each list will contain the ids of the products
        self.probes_arr = np.empty(number_of_probes, dtype=object)
        for i in xrange(number_of_probes):
            self.probes_arr[i] = []
        for i in xrange(n_samples):
            self.probes_arr[probe_serials[i]].append(i)

        self.probes_per_batch = probes_per_batch
        self.n_final_len = int(np.ceil(number_of_probes / probes_per_batch))
        self.build_final()

    def get_mini_batch_ids_internal(self, batch_id):
        """
        returns ids for batch_ids. this is serial, and until data is fixed for evry id unless shuffled.
        :param batch_id: the request batch_id.
        :return: the ids for this batch.
        """

        logging.debug("MiniBatchIdsByProbeId: get_mini_batch_ids: started.")
        product_ids = []
        for product_list in self.probes_arr[self.probes_per_batch * batch_id: self.probes_per_batch * (batch_id + 1)]:
            product_ids = (product_ids + product_list)
        product_ids = np.array(product_ids)
        # logging.info("MiniBatchIdsByProbeId: get_mini_batch_ids: product_ids: %s" % str(product_ids))
        logging.debug("MiniBatchIdsByProbeId: get_mini_batch_ids: Ended.")
        return product_ids

    def build_final(self):
        self.final = []
        for i in range(self.n_final_len):
            self.final = self.final + [self.get_mini_batch_ids_internal(i).tolist()]

    def get_mini_batch_ids(self, i):
        return self.final[i]


def read_from_disk(path):
    res = ocv.imread(path)
    res = res.astype('float32')
    res /= 255
    return res


def extract_featurs(feats_paths, ids):
    """
    :param feats_paths: paths to all products.
    :param ids: requested ids.
    :param task: the task this NN is performing
    :return: a tensor containing the feautures in desired format.
    """
    logging.debug("extract_featurs: Started.")
    logging.debug("there are %d examples. batch size is %d." % (len(feats_paths), len(ids)))
    feats = np.zeros((len(ids), product_width, product_height, product_channels), dtype=type(np.ndarray))
    for ii in range(len(ids)):
        res = read_from_disk(feats_paths[ids[ii]])
        feats[ii] = res
    ans = feats
    logging.debug("extract_featurs: Ended.")
    return ans


def create_mask_relation(rel_list):
    """

    :param rel_list: a relation list. size: n_nodes = |train|+|valid|+|test|. every node contains n_rels relations.
    :return:
    """
    logging.debug("create_mask_relation: Started.")
    n_nodes = len(rel_list)
    n_rels = len(rel_list[0])
    max_neigh = 0

    # go over all nodes, and all of their relation and find the largest relation.
    for node in rel_list:
        for rel in node:
            max_neigh = max(max_neigh, len(rel))

    # create arrays abel to handle the same amount of data, but for every relation it will hold the maximal amount
    # needed.
    rel = np.zeros((n_nodes, n_rels, max_neigh), dtype='int64')
    mask = np.zeros((n_nodes, 2, n_rels, max_neigh), dtype='float32')

    for i, node in enumerate(rel_list):  # go over source relation list.
        for j, r in enumerate(node):  # go over source relation.
            n = len(r)
            if n == 0:  # if the relation is empty
                mask[i, 1, j, 0] = 1
            else:
                rel[i, j, : n] = r  # copy the relation content.
                mask[i, 0, j, : n] = 1.0
                mask[i, 1, j, : n] = 1.0
    logging.debug("create_mask_relation: Ended.")
    return rel, mask

class SampleGenerator:
    def __init__(self, sample_index, sample_name,arg_dict,p, l, rl, rm, b, nested):
        logging.debug("SampleGenerator: constructor: Started. designed for sample index %d called: %s." %
                      (sample_index, sample_name))
        self.arg_dict = arg_dict
        self.p = p
        self.l = l
        self.rl = rl
        self.rm = rm


        self.curr_batch = 0
        self.si = sample_index
        self.name = sample_name
        self.nested_ids_array = nested[self.si]
        self.random_ids_gen = self.random_nested_ids_generator()
        self.train_ids_gen = MiniBatchIdsByProbeId(probe_serials=b[self.si], n_samples=len(b[self.si]),
                                                   number_of_probes=np.max(b[self.si]) + 1,
                                                   probes_per_batch=arg_dict['-batch_size'])
        self.max_batches = (np.max(b[self.si]) + 1) // arg_dict['-batch_size']
        self.n_samples = len(b[self.si])
        logging.debug("SampleGenerator: constructor: Ended")

    def prepare_data(self, ids, p, l, rl, rm):
        # sx = samples_x
        # sy = samples y
        # srm = samples relation mask
        # srl = samples relation list
        sx = [read_from_disk(path) for path in p[ids]]
        sy = l[ids]
        srm = rm[ids]
        srl = rl[ids]
        srl = np.subtract(srl, np.min(ids))
        srl = np.maximum(srl, 0)
        sx = np.array(sx)
        sy = np.array(sy)
        srl = np.array(srl)
        srm = np.array(srm)
        return sx, sy, srl, srm

    def random_nested_ids_generator(self):
        while True:
            random.shuffle(self.nested_ids_array)
            for arr in self.nested_ids_array:

                random.shuffle(arr)
                for product_id in arr:
                    yield product_id

    def data_generator(self):
        logging.info("SampleGenerator: %s has been started." % self.name)
        i = 0
        while True:
            if self.arg_dict['-batch_type'] == 'constant':
                ids = range(i, np.minimum(i + self.arg_dict['-constant_batch_size'], self.n_samples))
            elif self.arg_dict['-batch_type'] == 'random':
                ids = [self.random_ids_gen.next() for j in
                       range(i, np.minimum(i + self.arg_dict['-constant_batch_size'], self.n_samples))]
            elif self.arg_dict['-batch_type'] == 'probe':
                ids = self.train_ids_gen.get_mini_batch_ids(self.curr_batch)

            sx, sy, srl, srm = self.prepare_data(ids, self.p[self.si], self.l[self.si], self.rl[self.si], self.rm[self.si])
            self.curr_batch += 1
            self.curr_batch %= self.max_batches
            if self.arg_dict['-model_type'] == 'CNN':
                inp = [sx]
            else:
                inp = [sx, srl, srm]
            yield inp, sy
            i += self.arg_dict['-constant_batch_size']
            if i > self.n_samples:
                i = 0

    def get_ytrue(self):
        return self.l[self.si]


