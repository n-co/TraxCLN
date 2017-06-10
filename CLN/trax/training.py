from config import *

from prepare_data import *
from create_model import *
from callbacks import *

from keras.utils.visualize_util import plot as kplt

import datetime as dt

# GLOBAL VARIABLES
arg_dict = {}

def build_model(paths,rel_list):
    logging.debug("build_model: - Started")
    model_type = arg_dict['-model_type']
    dim = arg_dict['-dim']
    n_classes = arg_dict['-n_classes']
    nlayers = arg_dict['-nlayers']
    dropout = arg_dict['-dropout']
    fm = arg_dict['-flatmethod']
    shared = arg_dict['-shared']
    nmean = arg_dict['-nmean']
    example_x = extract_featurs(paths, [0])

    if model_type == 'HCNN':
        model = create_hcnn_relation(n_layers=nlayers, hidden_dim=dim, input_shape=example_x[0].shape,
                                     n_rel=rel_list.shape[-2],
                                     n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                                     dropout=dropout, flat_method=fm, pooling=pooling)
    elif model_type == 'CNN':
        model = create_cnn(input_shape=example_x[0].shape, n_classes=n_classes, pooling=pooling)

    all_optimizers = {
        'RMS2': rmsprop(lr=0.0001, decay=1e-6),
        'RMS': RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8),
        'Adam': Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    }
    selected_optimizer = all_optimizers[arg_dict['-opt']]

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=selected_optimizer,
                  metrics=['accuracy', 'fscore', 'precision', 'recall'])

    logging.debug("build_model: - Ended")
    logger.stop_and_read(run_mode)
    return model


def log_model(model,time_stamp):
    logging.debug("log_model: - Started.")
    saving = arg_dict['-saving']

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
    f.write('Training log:\n\n')

    for key in arg_dict:
        f.write('\t%-10s: %s\n' % (key, arg_dict[key]))

    f.write('\ninformation structure:\n')
    mn = ''
    for name in model.metrics_names:
        mn += '%-22s' % name
    f.write("time            epoch_id: ")
    f.write("valid: %s" % mn)
    f.write("test:  %s" % mn)
    f.write("\n")
    f.close()
    logging.debug("log_model: - Ended.")
    logger.stop_and_read(run_mode)
    return f_result, f_params


def log_summary(f_result,start_time, end_time):
    logging.debug("log_summary: - Started.")
    f = open(f_result, 'a')
    f.write('\n\n')
    f.write('start time:    %s\n' % start_time)
    f.write('end time:      %s\n' % end_time)
    f.write('total runtime: %s\n' % (end_time - start_time))
    f.close()
    logging.debug("log_summary: - Ended.")


def get_information(f_result,f_params):
    logging.debug("get_information: - Started.")
    performance_evaluator = Evaluator(file_result=f_result, file_params=f_params)

    callbacks = [performance_evaluator, NanStopping()]
    logging.debug("get_information: - Ended.")
    logger.stop_and_read(run_mode)
    return performance_evaluator, callbacks


def create_sub_samples(paths,labels,batches,rel_list,rel_mask,train_ids,valid_ids,test_ids):
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

    def create_nested_ids(ids):
        """
        creating a nested np array of product_ids, first dimension for a probe, second dimension is the product id
        :return: the nested array
        """
        logging.debug("create_nested_ids: - Started.")
        probes_offset = batches[ids[0]]
        products_offset = ids[0]
        num_of_probes = batches[ids[-1]] - probes_offset + 1
        nested_ids = np.empty((num_of_probes,), dtype=np.ndarray)

        products_list = []
        prev_probe_id = batches[ids[0]]
        for product_id in ids:
            probe_id = batches[product_id]
            if probe_id != prev_probe_id:
                nested_ids[prev_probe_id - probes_offset] = np.array(products_list)
                products_list = []
                prev_probe_id = probe_id
            products_list.append(product_id - products_offset)

        # add last list
        nested_ids[prev_probe_id - probes_offset] = np.array(products_list)

        logging.debug("create_nested_ids: - Ended.")
        return nested_ids

    inp = [train_ids, valid_ids, test_ids]
    nested = [create_nested_ids(train_ids),
              create_nested_ids(valid_ids),
              create_nested_ids(test_ids)]
    offsets = [0, len(train_ids), len(train_ids) + len(valid_ids)]
    p = [None, None, None]  # paths
    l = [None, None, None]  # labels
    rl = [None, None, None]  # rel list
    rm = [None, None, None]  # rel mask
    b = [None, None, None]  # batches
    for i in range(len(inp)):
        p[i], l[i], rl[i], rm[i], b[i] = project(inp[i], offsets[i], paths, labels, rel_list, rel_mask, batches)

    return p, l, rl, rm, b, nested




def main_cln():
    global arg_dict
    time_stamp = start_time = dt.datetime.now().replace(microsecond=0)
    logging.info('main: started. start time:    %s' % start_time)

    # calculates variables for execution.
    arg_dict, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, paths, batches = get_global_configuration(
        sys.argv)

    # build a keras model to be trained.
    model = build_model(paths,rel_list)

    # write information about model into log files.
    f_result, f_params = log_model(model, time_stamp)

    # get generators and additional variables.
    performance_evaluator, callbacks = get_information(f_result,f_params)

    # divide data into sub samples
    p, l, rl, rm, b, nested = create_sub_samples(paths,labels,batches,rel_list,rel_mask,train_ids,valid_ids,test_ids)

    logging.debug('main: running network...')
    train_gen = SampleGenerator(0,'train',arg_dict,p,l,rl,rm,b,nested)
    valid_gen = SampleGenerator(1, 'valid',arg_dict,p,l,rl,rm,b,nested)
    test_gen = SampleGenerator(2, 'test',arg_dict,p,l,rl,rm,b,nested)

    performance_evaluator.valid_gen = valid_gen
    performance_evaluator.test_gen = test_gen

    model.fit_generator(train_gen.data_generator(), samples_per_epoch=train_gen.n_samples, nb_epoch=number_of_epochs,
                        verbose=1, callbacks=callbacks,
                        validation_data=valid_gen.data_generator(), nb_val_samples=valid_gen.n_samples,
                        class_weight=None, max_q_size=10, nb_worker=1,
                        pickle_safe=False, initial_epoch=0, )

    end_time = dt.datetime.now().replace(microsecond=0)
    log_summary(start_time, end_time)
    logging.debug('main: training is complete.')
    logging.info('main: start time:    %s' % start_time)
    logging.info('main: end time:      %s' % end_time)
    logging.info('main: total runtime: %s' % (end_time - start_time))

    logging.debug('main: ended.')


main_cln()
