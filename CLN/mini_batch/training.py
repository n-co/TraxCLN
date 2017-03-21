from config import *
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


def build_model(model_type, n_layers, dim, example_x, rel_list, n_classes, shared, nmean,
                dropout, selected_optimizer, loss):
    logging.info("build_model: - Started")
    model = None
    if model_type == 'Highway':
        model = create_highway(n_layers=n_layers, hidden_dim=dim, input_dim=example_x.shape[-1],
                               n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                               n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
    elif model_type == 'Dense':
        model = create_dense(n_layers=n_layers, hidden_dim=dim, input_dim=example_x.shape[-1], n_rel=rel_list.shape[-2],
                             n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                             dropout=dropout)
    elif model_type == 'HCNN':
        model = create_hcnn(n_layers=n_layers, hidden_dim=dim, input_shape=example_x[0].shape, n_rel=rel_list.shape[-2],
                            n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                            dropout=dropout)
    elif model_type == 'Flat':
        model = create_flat(n_layers=n_layers, hidden_dim=dim, input_shape=example_x[0].shape, n_rel=rel_list.shape[-2],
                            n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean,
                            dropout=dropout)

    model.summary()
    # TODO: choose how to compile.
    # model.compile(optimizer=selected_optimizer, loss=loss)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    logging.info("build_model: - Ended")
    stop_and_read(run_mode)
    return model


def log_model(model, saving):
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


def get_information(paths, batches, n_layers, dim, rel_list, rel_mask, train_sample_size, batch_size, labels, valid_ids,
                    test_ids, task, f_result, f_params, model):
    logging.info("get_information: - Started.")
    # create a variable to hold hidden features passed through net so relations can be entered as inputs.
    hidden_data = HiddenSaving(n_samples=paths.shape[0], n_layers=n_layers, h_dim=dim, rel=rel_list, rel_mask=rel_mask)
    # create a variable containing a generator, that upon request generates a list of, possibly random, ids to select
    # from TRAIN SET.
    # mini_batch_ids_generator = MiniBatchIds(train_sample_size, batch_size=batch_size) #TODO: convert to generate by probe id
    mini_batch_ids_generator = MiniBatchIdsByProbeId(batches, train_sample_size, np.max(batches)+1, batch_size)
    # Exctract features and labels of validation and sample tests.
    valid_x = extract_featurs(paths, valid_ids, task)
    valid_y = labels[valid_ids]
    test_x = extract_featurs(paths, test_ids, task)
    test_y = labels[test_ids]
    # careate a callback to log information once a bacth is done.
    performence_evaluator = SaveResult(task=task, file_result=f_result, file_params=f_params)
    # collect all callbacks for model.
    callbacks = [performence_evaluator, NanStopping()]
    # get a simulations of functions in NN so to evaluate features of context.
    hidd_input_funcs = get_hidden_funcs_from_model(model, n_layers)
    logging.info("get_information: - Ended.")
    stop_and_read(run_mode)
    return hidden_data, mini_batch_ids_generator, valid_x, valid_y, test_x, test_y, performence_evaluator, \
        callbacks, hidd_input_funcs


def main_cln():
    # calculates variables for execution.
    dataset, task, model_type, n_layers, dim, shared, saving, nmean, batch_size, dropout, example_x, n_classes, loss, \
        selected_optimizer, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids, \
        paths, batches, train_sample_size, n_batchs = get_global_configuration(sys.argv)
    logging.debug(str(batches))
    # build a keras model to be trained.
    model = build_model(model_type, n_layers, dim, example_x, rel_list, n_classes, shared, nmean,
                        dropout, selected_optimizer, loss)


    # write information about model into log files.
    f_result, f_params = log_model(model, saving)

    # get generators and additional variables.
    hidden_data, mini_batch_ids_generator, valid_x, valid_y, test_x, test_y, performence_evaluator, callbacks, \
        hidd_input_funcs = get_information(paths, batches, n_layers, dim, rel_list, rel_mask, train_sample_size, batch_size,
                                           labels, valid_ids, test_ids, task, f_result, f_params, model)

    logging.info("CLN - Started.")
    stop_and_read(run_mode)
    for epoch in xrange(number_of_epochs):  # train the network a few times to get more accurate results.
        start = time.time()
        for i in xrange(n_batchs):  # go over all batches. all of training data will be used.
            batch_start = time.time()

            # Extract features and labels of TRAIN set.
            logging.debug("batch number req is: %d" % i)
            stop_and_read(run_mode)
            mini_batch_ids = train_ids[mini_batch_ids_generator.get_mini_batch_ids(i)]
            train_x = extract_featurs(paths, mini_batch_ids, task)
            train_y = labels[mini_batch_ids]
            train_rel_list = rel_list[mini_batch_ids]
            train_rel_mask = rel_mask[mini_batch_ids]
            model.fit([train_x,train_rel_list], numpy.expand_dims(train_y, -1), validation_data=valid_data, verbose=0,
                      nb_epoch=1, batch_size=train_x.shape[0], shuffle=False, callbacks=callbacks)


            # get hidden context for train set.
            train_context = hidden_data.get_context(mini_batch_ids)
            learn_about(train_context, run_mode)

            # in the last mini-batch of an epoch compute the context for valid and test, save result.
            # update hidden states of valid and test in past_hiddens.
            if i == n_batchs - 1:
                # get hidden context for valid set.
                valid_context = hidden_data.get_context(valid_ids)

                # get hidden context for test set.
                test_context = hidden_data.get_context(test_ids)

                # performence evaluator is activated at the end on an epoch. this makes sure it is up to date.
                # note that form one epoch to the other, the context change, not the features.
                performence_evaluator.update_data([valid_x] + valid_context, valid_y, [test_x] + test_context, test_y)

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
            batch_end = time.time()
            logging.info("batch %d in epoch %d. is done. time: %f." % (i, epoch, batch_end - batch_start))
        end = time.time()
        logging.info('epoch %d, runtime: %.1f' % (epoch, end - start))
        if model.stop_training:
            break
    logging.info("CLN - Ended.")

main_cln()
