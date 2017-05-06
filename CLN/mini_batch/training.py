from config import *
import random
import numpy
import sys
import time
from prepare_data import *
from create_model import *
from callbacks import *
from app_hidden import *
from keras.optimizers import *
from keras.objectives import *
from keras.utils.visualize_util import plot as kplt

# GLOBAL VARIABLES
dataset = task = n_layers = dim = shared = saving = nmean = batch_size = dropout = example_x = n_classes = loss = selected_optimizer = batch_type = fm = model_type = None
paths = labels = batches = rel_list = rel_mask = None
train_ids = valid_ids = test_ids = None
train_sample_size = n_batchs = None
hidden_data = hidd_input_funcs = train_mini_batch_ids_generator = None
valid_x = valid_y = test_x = test_y = None
f_result = f_params = None
model = performance_evaluator = callbacks = None
p = l = rl = rm = b = None
train_gen = valid_gen = test_gen = None


def build_model():
    logging.info("build_model: - Started")
    global model
    model = None
    if model_type == 'Highway':
        model = create_highway(n_layers=n_layers, hidden_dim=dim, input_dim=example_x.shape[-1],
                               n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                               n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
    elif model_type == 'Dense':
        model = create_dense(n_layers=n_layers, hidden_dim=dim, input_dim=example_x.shape[-1],
                             n_rel=rel_list.shape[-2],
                             n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                             dropout=dropout)
    elif model_type == 'HCNN' and batch_type == "context":
        model = create_hccn_context(n_layers=n_layers, hidden_dim=dim, input_shape=example_x[0].shape,
                                    n_rel=rel_list.shape[-2],
                                    n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                                    dropout=dropout, flat_method=fm)
    elif model_type == 'HCNN' and batch_type == "relation":
        model = create_hcnn_relation(n_layers=n_layers, hidden_dim=dim, input_shape=example_x[0].shape,
                                     n_rel=rel_list.shape[-2],
                                     n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                                     dropout=dropout, flat_method=fm)
    elif model_type == 'CNN':
        model = create_cnn(input_shape=example_x[0].shape,n_classes=n_classes)

    # model.summary()
    # model.compile(optimizer=selected_optimizer, loss=loss)

    opt = rmsprop(lr=0.0001, decay=1e-6)
    model.summary()
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    logging.info("build_model: - Ended")

    stop_and_read(run_mode)
    return model


def log_model():
    global f_params, f_result
    logging.info("log_model: - Started.")
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
    f_result = logs_path + saving + '.txt'
    f = open(f_result, 'w')
    f.write('Training log:\n')
    f.write('information structure:\n')
    f.write('e#: epoch_id\n')
    f.write('\tvalidation:\n')
    f.write('\t\tloss\tval_los\t|\tv_auc\tv_f1\tv_pre\tv_rec\n')
    f.write('\t\tv_auc\tv_err\n')
    f.write('\ttest:\n')
    f.write('\t\tt_auc\tt_f1\tt_pre\tt_rec\n')
    f.write('\t\tt_acc\tt_err\n')
    f.close()
    logging.info("log_model: - Ended.")
    stop_and_read(run_mode)
    return f_result, f_params


def get_information():
    global hidden_data, train_mini_batch_ids_generator, valid_x, valid_y, test_x, test_y, \
        performance_evaluator, callbacks, hidd_input_funcs

    logging.info("get_information: - Started.")
    # create a variable to hold hidden features passed through net so relations can be entered as inputs.


    # careate a callback to log information once a bacth is done.

    # collect all callbacks for model.

    # get a simulations of functions in NN so to evaluate features of context.
    if batch_type == 'context':
        performence_evaluator = SaveResult(task=task, file_result=f_result, file_params=f_params)
        callbacks = [performence_evaluator, NanStopping()]
        hidd_input_funcs = get_hidden_funcs_from_model(model, n_layers)
        train_mini_batch_ids_generator = MiniBatchIds(train_sample_size,
                                                      batch_size=batch_size)
        # Exctract features and labels of validation and sample tests.
        valid_x = extract_featurs(paths, valid_ids, task)
        valid_y = labels[valid_ids]
        test_x = extract_featurs(paths, test_ids, task)
        test_y = labels[test_ids]
        hidden_data = HiddenSaving(n_samples=paths.shape[0], n_layers=n_layers, h_dim=dim, rel=rel_list,
                                   rel_mask=rel_mask)
    else:
        performence_evaluator = Evaluator(test_gen=None,file_result=f_result, file_params=f_params)
        callbacks = [performence_evaluator, NanStopping()]
    logging.info("get_information: - Ended.")
    stop_and_read(run_mode)
    return hidden_data, train_mini_batch_ids_generator, valid_x, valid_y, test_x, test_y, performence_evaluator, \
           callbacks, hidd_input_funcs


def create_sub_samples():
    global paths,labels,rel_list,batches
    global train_ids,valid_ids,test_ids
    global p, l, rl, rm, b

    def project(indexes, paths, labels, rel_list, rel_mask, batches, offset):
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
    p = [None, None, None]
    l = [None, None, None]
    rl = [None, None, None]
    rm = [None, None, None]
    b = [None, None, None]
    for i in range(len(inp)):
        p[i], l[i], rl[i], rm[i], b[i] = project(inp[i], paths, labels, rel_list, rel_mask, batches, offsets[i])


def run_nn_context(i):
    mini_batch_ids = train_ids[train_mini_batch_ids_generator.get_mini_batch_ids(i)]
    train_x = extract_featurs(paths, mini_batch_ids, task)
    train_y = labels[mini_batch_ids]

    # get hidden context for train set.
    train_context = hidden_data.get_context(mini_batch_ids)
    # learn_about(train_context, run_mode)

    # in the last mini-batch of an epoch compute the context for valid and test, save result.
    # update hidden states of valid and test in past_hiddens.
    if i == n_batchs - 1:
        # get hidden context for valid set.
        valid_context = hidden_data.get_context(valid_ids)

        # get hidden context for test set.
        test_context = hidden_data.get_context(test_ids)

        # performence evaluator is activated at the end on an epoch. this makes sure it is up to date.
        # note that form one epoch to the other, the context change, not the features.
        performance_evaluator.update_data([valid_x] + valid_context, valid_y, [test_x] + test_context, test_y)

        # this loop simply wrapps lists into tuples, and allow adressing each member in the tuple with a name.
        # this loop actually runs twice in each epoch: once for valid, once for test.
        # in each iterations, hidden data is updated for those sets.
        for x, ct, ids in zip([valid_x, test_x], [valid_context, test_context], [valid_ids, test_ids]):
            new_hiddens = calc_hidden(x, ct, hidd_input_funcs)
            hidden_data.update_hidden(ids, new_hiddens)

        # define validation data.
        valid_data = [[valid_x] + valid_context, valid_y]
    else:
        valid_data = None

    # train the model
    model.fit([train_x] + train_context, numpy.expand_dims(train_y, -1), validation_data=valid_data, verbose=0,
              nb_epoch=1, batch_size=train_x.shape[0], shuffle=False, callbacks=callbacks)

    # update hidden data for train set.

    new_train_hiddens = calc_hidden(train_x, train_context, hidd_input_funcs)
    hidden_data.update_hidden(mini_batch_ids, new_train_hiddens)


class SampleGenerator():

    # global batch_size,task

    def __init__(self, sample_index, sample_name):
        logging.info("SampleGenerator: constructor: Started. designed for sample index %d called: %s." % (sample_index, sample_name))
        self.curr_batch = 0
        self.si = sample_index
        self.name = sample_name
        self.train_ids_gen = MiniBatchIdsByProbeId(probe_serials=b[self.si], n_samples=len(b[self.si]),
                                                   number_of_probes=np.max(b[self.si]) + 1,
                                                   probes_per_batch=batch_size)
        self.max_batches = (np.max(b[self.si]) + 1) //  batch_size
        self.n_samples = len(b[self.si])
        logging.info("SampleGenerator: constructor: Ended")

    def read_from_disk(self,path):
        res = ocv.imread(path)
        res = res.astype('float32')
        res /= 255
        return res

    def prep(self,ids, p, l, rl, rm):
        sx= [self.read_from_disk(ppp) for ppp in p[ids]]
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


    def generator_creator(self):
        bc = 128
        i = 0
        while True:
            # ids = range(i,np.minimum(i+bc,len(p[self.si])))
            ids = self.train_ids_gen.get_mini_batch_ids(self.curr_batch)
            sx,sy,srl,srm = self.prep(ids,p[self.si],l[self.si],rl[self.si],rm[self.si])
            i += bc
            i %= len(p[self.si])
            self.curr_batch +=1
            self.curr_batch %=self.max_batches
            if model_type == 'CNN':
                inp = [sx]
            else:
                inp = [sx,srl,srm]
            yield inp,sy


    def prepare_batch(self,ids, p, l, rl, rm):
        sx = extract_featurs(p, ids, task)
        sy = l[ids]
        srm = rm[ids]
        srl = rl[ids]
        srl = np.subtract(srl, np.min(ids))
        srl = np.maximum(srl, 0)
        return sx, sy, srl, srm

    def wrap_with_generator(self,z):
        while True:
            yield z[0], z[1]

    def batches_iterator(self,i):
        while self.curr_batch < self.max_batches:
            ids = self.train_ids_gen.get_mini_batch_ids(i)
            logging.debug("generating batch for sample: " + self.name + " curr batch: " + str(self.curr_batch) + " out of:  " + str(self.max_batches) + " ids generated: " + str(ids))
            if ids is not []:
                sx, sy, srl, srm = self.prepare_batch(ids, p[self.si], l[self.si], rl[self.si], rm[self.si])
                # stop_and_read('debug')
            self.curr_batch += 1
            if model_type=='CNN':
                inp = sx
            else:
                inp = [sx,srl,srm]
            return [inp, sy]

    def batches_generator(self):
        while True:
            ids = self.train_ids_gen.get_mini_batch_ids(self.curr_batch)
            logging.debug("generating batch for sample: " + self.name + " " + str(self.curr_batch) + "  " + str(self.max_batches) + "   " + str(ids))
            if ids != np.array([]):
                sx, sy, srl, srm = self.prepare_batch(ids, p[self.si], l[self.si], rl[self.si], rm[self.si])
                # stop_and_read('debug')
            self.curr_batch += 1
            if model_type == 'CNN':
                inp = sx
            else:
                inp = [sx,srl,srm]
            if ids != np.array([]):
                if self.name != 'test':
                    yield inp, sy
                else:
                    yield inp
            else:
                logging.error("overhead!")
                self.curr_batch = 0

    def build_gen(self):
        self.curr_batch =0
        self.gen = self.batches_generator()

    def get_ytrue(self):
        return l[self.si]


def run_nn_relation(i):
    global train_gen,valid_gen,test_gen

    logging.info("run_nn_relation: started")
    train_batch_data = train_gen.batches_iterator(i)
    train_size = train_batch_data[1].shape[0]
    train_batch_generator = train_gen.wrap_with_generator(train_batch_data)

    valid_batch_generator = None
    valid_size = None
    local_callbacks = None

    if (i+1) % (batch_size*5+1) == 0:
        valid_gen.build_gen()
        valid_batch_generator = valid_gen.gen
        valid_size = valid_gen.n_samples
        logging.debug("validation sample size: %d." % valid_size)
        test_gen.build_gen()
        performance_evaluator.set_gen(test_gen)
        # local_callbacks = callbacks

    model.fit_generator(train_batch_generator, samples_per_epoch=train_size, nb_epoch=1,
                        verbose=1, callbacks=local_callbacks,
                        validation_data=valid_batch_generator, nb_val_samples=valid_size,
                        class_weight=None,
                        max_q_size=10, nb_worker=1, pickle_safe=False,
                        initial_epoch=0)
    logging.info("run_nn_relation: Ended")



def main_cln():
    global dataset , task , n_layers , dim , shared , saving , nmean , batch_size , dropout , example_x , n_classes , loss , selected_optimizer , batch_type , fm , model_type
    global paths , labels , batches , rel_list , rel_mask
    global train_ids , valid_ids , test_ids
    global train_sample_size , n_batchs
    global hidden_data , hidd_input_funcs , train_mini_batch_ids_generator
    global valid_x , valid_y , test_x , test_y
    global f_result , f_params
    global model , performance_evaluator , callbacks
    global p , l , rl , rm , b
    global train_gen , valid_gen , test_gen


    # calculates variables for execution.
    dataset, task, model_type, n_layers, dim, shared, saving, nmean, batch_size, dropout, example_x, n_classes, loss, \
    selected_optimizer, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, \
    paths, batches, train_sample_size, n_batchs, batch_type, fm = get_global_configuration(sys.argv)

    # build a keras model to be trained.
    build_model()

    # write information about model into log files.
    log_model()

    # get generators and additional variables.
    get_information()

    # devide data into sub samples
    create_sub_samples()

    logging.info('PRECLN: started.')
    train_gen = SampleGenerator(sample_index=0, sample_name='train')
    valid_gen = SampleGenerator(sample_index=1, sample_name='valid')
    test_gen = SampleGenerator(sample_index=2, sample_name='test')
    train_gen.build_gen()
    valid_gen.build_gen()
    test_gen.build_gen()
    model.fit_generator(train_gen.generator_creator(), samples_per_epoch=train_gen.n_samples, nb_epoch=10,

                        verbose=1, callbacks=None,
                        validation_data=test_gen.generator_creator(), nb_val_samples=test_gen.n_samples,
                        class_weight=None, max_q_size=10, nb_worker=1,
                        pickle_safe=False, initial_epoch=0, )
    logging.info('PRECLN: ended.')
    exit()
    logging.info("CLN - Started.")
    stop_and_read(run_mode)
    for epoch in xrange(1, number_of_epochs):  # train the network a few times to get more accurate results.
        start = time.time()
        if batch_type=='relation':
            train_gen = SampleGenerator(sample_index=0, sample_name='train')
            valid_gen = SampleGenerator(sample_index=1, sample_name='valid')
            test_gen = SampleGenerator(sample_index=2, sample_name='test')
        for i in xrange(n_batchs):  # go over all batches. all of training data will be used.
            batch_start = time.time()
            if batch_type == 'context':
                run_nn_context(i)
            elif batch_type == 'relation':
                run_nn_relation(i)
            batch_end = time.time()
            logging.info("batch %d in epoch %d. is done. time: %f." % (i, epoch, batch_end - batch_start))
        end = time.time()
        logging.info('epoch %d, runtime: %.1f' % (epoch, end - start))
        if model.stop_training:
            break
    logging.info("CLN - Ended.")

main_cln()
