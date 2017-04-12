from config import *
import gzip
import cPickle
import numpy
from keras.optimizers import *
from keras.objectives import *
from keras.utils.np_utils import to_categorical


def process_input_args(argv):
    """
    a function to process input arguments.
    :param argv: the command line args that were passed.
    :return: arg_dict: a json dictionary containing args in the desire.
    """
    logging.info("process_input_args: Started.")
    i = 1
    # set default args.
    arg_dict = {
        '-data': 'pubmed',  # chosing the learning data set: trax, pubmed, movielens, software.
        '-saving': 'pubmed',  # log file name.
        '-model': '',  # type of NN for each column network. Highway/Dence/
        '-batch': 100,  # batch size for mini-batch version.
        '-y': 1,  # incase of multilabel training, decides on which one net is trained.
        '-nlayers': 10,  # number of layers in each highway network.
        '-dim': 50,  # number of nodes in each layer of each coloumn network.
        '-shared': 1,  # indicator. 1: parameters will be shared between coloumns. 0: no sharing.
        '-nmean': 1,  # regulzation factor. shoule be 1<=nmean<=number_of_relations
        '-reg': '',  # indicator: dr: dropout. nothing: no dropout.
        '-opt': 'RMS',  # or Adam. an optimizer for paramater tuning.
        '-seed': 1234,  # used to make random decisions repeat.
        '-trainlimit': -1,  # a limitation on the size of train set.
        '-batchtype': 'context',  # 'context' / 'relation'. changes architecture: mini or full batch.
        '-flatmethod': 'c'  # 'c' -CNN. 'f' - Flat. else - No Flat.
    }
    # Update args to contain the user's desired configuration.
    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i + 1]
        i += 2
    # Update data types for arguments.
    arg_dict['-nlayers'] = int(arg_dict['-nlayers'])
    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-shared'] = int(arg_dict['-shared'])
    arg_dict['-nmean'] = int(arg_dict['-nmean'])
    arg_dict['-seed'] = int(arg_dict['-seed'])
    arg_dict['-y'] = int(arg_dict['-y'])
    arg_dict['-trainlimit'] = int(arg_dict['-trainlimit'])
    logging.info("process_input_args: Ended.")
    return arg_dict


def get_global_configuration(argv):
    """

    :param argv: the command line args that were passed.
    :return: many variables to be used as globals in main.
    """
    logging.info("get_global_configuration - Started.")
    args = process_input_args(argv)
    seed = args['-seed']
    numpy.random.seed(seed)
    dataset = args['-data']
    task = 'software'
    if 'pubmed' in dataset:
        task = 'pubmed'
    elif 'movie' in dataset:
        task = 'movie'
    elif 'trax' in dataset:
        task = 'trax'
    dataset = data_sets_dir + dataset + '.pkl.gz'
    model_type = args['-model']
    n_layers = args['-nlayers']
    dim = args['-dim']
    shared = args['-shared']
    saving = args['-saving']
    nmean = args['-nmean']
    yidx = args['-y']
    batch_size = int(args['-batch'])
    train_limit = args['-trainlimit']
    batch_type = args['-batchtype']
    fm = args['-flatmethod']

    if 'dr' in args['-reg']:
        dropout = True
    else:
        dropout = False

    if task == 'trax':
        labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, paths, batches = load_data_trax(dataset,batch_type)
    else:
        feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids = load_data(dataset)
        paths = feats
        batches = feats #TODO: this is dummy wo it will be of correct data type

    example_x = extract_featurs(paths, [0], task)

    if train_limit == -1:
        train_sample_size = len(train_ids)
    else:
        train_sample_size = train_limit

    labels = labels.astype('int64')


    # the number of classes is max+1 since the first class is 0.
    # in binary classification, this parameter is probably not used.
    n_classes = numpy.max(labels)
    if n_classes > 1:
        n_classes += 1
        loss = sparse_categorical_crossentropy
    else:
        loss = binary_crossentropy

    labels = to_categorical(labels, n_classes)

    # define the optimizer for weights finding.
    all_optimizers = {
        'RMS': RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8),
        'Adam': Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    }
    selected_optimizer = all_optimizers[args['-opt']]

    if batch_type =='context':
        n_batchs = train_sample_size // batch_size
    else:
        n_batchs = np.max(batches) + 1
    if train_sample_size % batch_size > 0:
        n_batchs += 1

    logging.info("get_global_configuration - Ended.")
    stop_and_read(run_mode)
    return dataset, task, model_type, n_layers, dim, shared, saving, nmean, batch_size, dropout, example_x, n_classes, \
           loss, selected_optimizer, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, \
           paths, batches, train_sample_size, n_batchs, batch_type, fm


def create_mask_context(rel_list):
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


def load_data_trax(path,batchtype):
    """
    loads data from a pcl file into memory.
    :param: path: full path to a gzip file, containing cPickle data.
    :return: content of cPickle data, in seperate arrays all of the size.
            this means, for every arr returned, a.shape[0] is the same
    """
    logging.info("load_data - Started.")
    f = gzip.open(path, 'rb')
    labels, rel_list, train_ids, valid_ids, test_ids, paths, batches = cPickle.load(f)
    logging.debug(str(paths))

    if batchtype == 'relation':
        rel_list, rel_mask = create_mask_relation(rel_list)
    elif batchtype=='context':
        rel_list, rel_mask = create_mask_context(rel_list)
    logging.info("load_data - Ended.")
    return labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, paths, batches


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
    rel_list, rel_mask = create_mask_context(rel_list)
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
        :return: an array of indexes, continious, describing ids of the burrent batch.
        :operations: when batch_id is 0, the ids array is suffled. this happens at the begining of every epoch, so every
                     epoch covers all train ids, but in a different order.
        """
        # TODO: implement this in other version.
        if batch_id == 0:
            numpy.random.shuffle(self.ids)
        return self.ids[self.batch_size * batch_id: self.batch_size * (batch_id + 1)]


class MiniBatchIdsByProbeId:
    # TODO: add suffel whenever a new epoch starts.
    def __init__(self, probe_serials, n_samples, number_of_probes, probes_per_batch):
        '''
        designed to handle a single Sample!(train/valid/test).
        :param probe_serials: a list of length 'n_samples'. the nth element in the list is a serial of the probe that
         contains the nth product (bottle) in the csv file.
        :param n_samples: number of products
        :param number_of_probes:
        :param probes_per_batch:
        '''
        logging.info("MiniBatchIdsByProbeId constructor")
        logging.info("can handle: probes: %d. products: %d. probes in batch: %d." %(n_samples,number_of_probes,probes_per_batch))
        # probes_arr = a numpy array of lists. each list will contain the ids of the products
        self.probes_arr = np.empty(number_of_probes, dtype=object)
        for i in xrange(number_of_probes):
            self.probes_arr[i] = []
        for i in xrange(n_samples):
            self.probes_arr[probe_serials[i]].append(i)

        self.probes_per_batch = probes_per_batch

    def get_mini_batch_ids(self, batch_id):
        """
        returns ids for batch_ids. this is serial, and until data is fixed for evry id unless shuffled.
        :param batch_id: the request batch_id.
        :return: the ids for this batch.
        """
        # if batch_id == 0:
        #     numpy.random.shuffle(self.probes_arr)
        #     for i in xrange(len(self.probes_arr)):
        #         numpy.random.shuffle(self.probes_arr[i])

        logging.info("MiniBatchIdsByProbeId: get_mini_batch_ids: started.")
        product_ids = []
        for product_list in self.probes_arr[self.probes_per_batch * batch_id: self.probes_per_batch * (batch_id + 1)]:
            product_ids = (product_ids + product_list)
        product_ids = np.array(product_ids)
        # logging.debug("MiniBatchIdsByProbeId: get_mini_batch_ids: product_ids: %s" % str(product_ids))
        logging.info("MiniBatchIdsByProbeId: get_mini_batch_ids: Ended.")
        return product_ids


def extract_featurs(feats_paths, ids, task):
    """
    :param feats_paths: paths to all products.
    :param ids: requested ids.
    :param task: the task this NN is performing
    :return: a tensor containing the feautures in desired format.
    """
    logging.info("extract_featurs: Started.")
    logging.info("there are %d examples. batch size is %d. task is %s" % (len(feats_paths), len(ids), task))
    feats = np.zeros((len(ids), product_width, product_height, product_channels), dtype=type(np.ndarray))
    if task == 'trax':
        for ii in range(len(ids)):
            res = ocv.imread(feats_paths[ids[ii]])
            res = res.astype('float32')
            res /= 255
            feats[ii] = res
        ans = feats
    else:
        ans = feats_paths[ids]
    logging.info("extract_featurs: Ended.")
    return ans


def create_mask_relation(rel_list):
    """

    :param rel_list: a relation list. size: n_nodes = |train|+|valid|+|test|. every node contains n_rels relations.
    :return:
    """
    logging.info("create_mask_relation: Started.")
    n_nodes = len(rel_list)
    n_rels = len(rel_list[0])
    max_neigh = 0

    #go over all nodes, and all of their relation and find the largest relation.
    for node in rel_list:
        for rel in node:
            max_neigh = max(max_neigh, len(rel))

    #create arrays abel to handle the same amount of data, but for every relation it will hold the maximal amount
    # needed.
    rel = numpy.zeros((n_nodes, n_rels, max_neigh), dtype='int64')
    mask = numpy.zeros((n_nodes, 2, n_rels, max_neigh), dtype='float32')

    for i, node in enumerate(rel_list): #go over source relation list.
        for j, r in enumerate(node): # go over source relation.
            n = len(r)
            if n == 0:  # if the relation is empty
                mask[i, 1, j, 0] = 1
            else:
                rel[i, j, : n] = r   #copy the relation content.
                mask[i, 0, j, : n] = 1.0
                mask[i, 1, j, : n] = 1.0
    logging.info("create_mask_relation: Ended.")
    return rel, mask
