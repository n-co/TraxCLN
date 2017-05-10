from config import *
import random
import numpy
import sys
import time
import datetime as dt
from prepare_data import *
from create_model import *
from callbacks import *
from keras.optimizers import *
from keras.objectives import *
from keras.utils.visualize_util import plot as kplt
from random import  randint


# GLOBAL VARIABLES
dataset = task = n_layers = dim = shared = saving = nmean = batch_size = dropout = example_x = n_classes = selected_optimizer = fm = model_type = None
paths = labels = batches = rel_list = rel_mask = None
train_ids = valid_ids = test_ids = None
f_result = f_params = None
model = performance_evaluator = callbacks = None
p = l = rl = rm = b = None
train_gen = valid_gen = test_gen = None
time_stamp = None


def build_model():
    logging.debug("build_model: - Started")
    global model
    model = None
    if model_type == 'HCNN':
        model = create_hcnn_relation(n_layers=n_layers, hidden_dim=dim, input_shape=example_x[0].shape,
                                     n_rel=rel_list.shape[-2],
                                     n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                                     dropout=dropout, flat_method=fm)
    elif model_type == 'CNN':
        model = create_cnn(input_shape=example_x[0].shape, n_classes=n_classes)


    model.summary()
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=selected_optimizer,
                  metrics=['accuracy','fscore','precision','recall'])

    logging.debug("build_model: - Ended")

    stop_and_read(run_mode)
    return model


def log_model():
    global f_params, f_result
    logging.debug("log_model: - Started.")
    # Log information so far.
    # Prints the model, in a json format, to the desired path.
    json_string = model.to_json()
    f_model = models_path + saving + '.json'
    f = open(f_model, 'w')
    f.write(json_string)
    f.close()

    kplt(model, to_file=models_path + saving + '.png', show_shapes=True)

    # Define path for saving results.
    f_params = best_models_path + saving + '.hdf5'

    # Create a log.
    time_str = time_stamp.strftime("%Y-%m-%d_%H-%M-%S")
    f_result = logs_path + saving + '__' + time_str + '.txt'
    f = open(f_result, 'w')
    f.write('Training log:\n')
    f.write('information structure:\n')
    # mn = str(model.metrics_names)
    mn = ''
    for name in model.metrics_names:
        mn += '%-22s' % name
    f.write("time            epoch_id: ")
    # f.write("train: %s" % mn)
    f.write("valid: %s" % mn)
    f.write("test:  %s" % mn)
    f.write("\n")
    f.close()
    logging.debug("log_model: - Ended.")
    stop_and_read(run_mode)
    return f_result, f_params


def log_summary(start_time, end_time):
    f = open(f_result, 'w')
    f.write('\n\n')
    f.write('start time:    %s\n' % start_time)
    f.write('end time:      %s\n' % end_time)
    f.write('total runtime: %s\n' % (end_time - start_time))
    f.close()


def get_information():
    global performance_evaluator, callbacks
    logging.debug("get_information: - Started.")
    performance_evaluator = Evaluator(file_result=f_result, file_params=f_params)

    callbacks = [performance_evaluator, NanStopping()]
    # callbacks = None
    logging.debug("get_information: - Ended.")
    stop_and_read(run_mode)
    return performance_evaluator, callbacks


def create_sub_samples():
    global paths, labels, rel_list, batches
    global train_ids, valid_ids, test_ids
    global p, l, rl, rm, b

    def project(indexes, offset, paths, labels, rel_list, rel_mask, batches):
        p = paths[indexes]
        l = labels[indexes]
        rl = rel_list[indexes]
        rl = np.subtract(rl, offset)
        rl = np.maximum(rl, 0)
        rm = rel_mask[indexes]
        b = batches[indexes]
        b = np.subtract(b, np.min(b))

        return p, l, rl, rm, b

    inp = [train_ids, valid_ids, test_ids]
    offsets = [0, len(train_ids), len(train_ids) + len(valid_ids)]
    p = [None, None, None]   # paths
    l = [None, None, None]   # labels
    rl = [None, None, None]  # rel list
    rm = [None, None, None]  # rel mask
    b = [None, None, None]   # batches
    for i in range(len(inp)):
        p[i], l[i], rl[i], rm[i], b[i] = project(inp[i],offsets[i], paths, labels, rel_list, rel_mask, batches)


class SampleGenerator:
    def __init__(self, sample_index, sample_name):
        logging.debug("SampleGenerator: constructor: Started. designed for sample index %d called: %s." %
                      (sample_index, sample_name))
        self.curr_batch = 0
        self.si = sample_index
        self.name = sample_name
        self.train_ids_gen = MiniBatchIdsByProbeId(probe_serials=b[self.si], n_samples=len(b[self.si]),
                                                   number_of_probes=np.max(b[self.si]) + 1,
                                                   probes_per_batch=batch_size)
        self.max_batches = (np.max(b[self.si]) + 1) // batch_size
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

    def data_generator(self):
        logging.debug("SampleGenerator: %s has been started." % self.name)
        # i and bs are used for constant batch sizes.
        # it is much faster than changing batch size.
        i = 0
        bs = 128
        while True:
            # ids = self.train_ids_gen.get_mini_batch_ids(self.curr_batch)
            ids = range(i, np.minimum(i+bs, self.n_samples))
            i += bs
            if i > self.n_samples: i = 0
            sx, sy, srl, srm = self.prepare_data(ids, p[self.si], l[self.si], rl[self.si], rm[self.si])
            self.curr_batch += 1
            self.curr_batch %= self.max_batches
            if model_type == 'CNN':
                inp = [sx]
            else:
                inp = [sx, srl, srm]
            yield inp, sy

    def get_ytrue(self):
        return l[self.si]


def main_cln():
    global dataset, task, n_layers, dim, shared, saving, nmean, batch_size, dropout, example_x, n_classes, loss, selected_optimizer, batch_type, fm, model_type
    global paths, labels, batches, rel_list, rel_mask
    global train_ids, valid_ids, test_ids
    global f_result, f_params
    global model, performance_evaluator, callbacks
    global p, l, rl, rm, b
    global train_gen, valid_gen, test_gen
    global time_stamp

    time_stamp = start_time = dt.datetime.now().replace(microsecond=0)
    print 'start time:    %s' % start_time

    # calculates variables for execution.
    dataset, task, model_type, n_layers, dim, shared, saving, nmean, batch_size, dropout, example_x, n_classes, \
        selected_optimizer, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, \
        paths, batches, fm = get_global_configuration(sys.argv)

    # build a keras model to be trained.
    build_model()

    # write information about model into log files.
    log_model()

    # get generators and additional variables.
    get_information()

    # devide data into sub samples
    create_sub_samples()

    logging.debug('PRECLN: started.')
    train_gen = SampleGenerator(sample_index=0, sample_name='train')
    valid_gen = SampleGenerator(sample_index=1, sample_name='valid')
    test_gen = SampleGenerator(sample_index=2, sample_name='test')

    # performence_evaluator.train_gen = train_gen
    performance_evaluator.valid_gen = valid_gen
    performance_evaluator.test_gen = test_gen

    # callbacks = None

    model.fit_generator(train_gen.data_generator(), samples_per_epoch=train_gen.n_samples, nb_epoch=number_of_epochs,
                        verbose=1, callbacks=callbacks,
                        validation_data=valid_gen.data_generator(), nb_val_samples=valid_gen.n_samples,
                        class_weight=None, max_q_size=10, nb_worker=1,
                        pickle_safe=False, initial_epoch=0, )

    end_time = dt.datetime.now().replace(microsecond=0)

    log_summary(start_time, end_time)

    print 'start time:    %s' % start_time
    print 'end time:      %s' % end_time
    print 'total runtime: %s' % (end_time - start_time)

    logging.debug('PRECLN: ended.')

main_cln()
