from config import *
import numpy
# from theano import tensor
import tensorflow as tensor
from keras.callbacks import *
from keras import objectives
from keras.layers import *
from keras.models import Model
from sklearn import metrics

from keras.constraints import *
from keras.layers.advanced_activations import *
from graph_layers import *


class SaveResult(Callback):
    """
    inherits from Callback.
    Compute result after each epoch. Return a log of result
    Arguments:
        data_x, data_y, metrics
    """

    def __init__(self, task='software', file_result='', file_params='', min_patience=5, max_patience=20):
        """
        a call back to be called when an epoch is ended. definde many empty vars on its instance.
        :param task: the task that this callback performs: trax, pubmed etc.
        :param file_result: full path to log file.
        :param file_params: full path to file saving the parameters the network has found.
        :param min_patience: a parameter regarding learning rate.
        :param max_patience: a parameter regarding learning rate.
        :operations: defines member variables for class.
        """
        super(SaveResult, self).__init__()  # call constructor of Callback Class.

        self.valid_x_d_c = None
        self.valid_y = None
        self.test_x_d_c = None
        self.test_y = None
        self.do_test = False
        self.save_result = False

        self.bestResult = 0.0
        self.bestEpoch = 0
        self.n_epoch = -1
        # wait to divide the learning rate. if reach the maxPatience -> stop learning
        self.wait = 0
        self.minPatience = min_patience
        self.maxPatience = max_patience

        self.task = task
        self.fileResult = file_result
        self.fileParams = file_params

    def update_data(self, valid_x_d_c, valid_y, test_x_d_c=None, test_y=None):
        """
        saves values on local instance. this will be used in the future.
        invoked on epoch end: last mini batch of an epoch. test_x is never none.
        note that save_result only becomes true when this method is invoked. it is invoked in the last mini batch.
        this is when save_result becomes true, and then imedietlry it will be false again.
        :param valid_x_d_c: features of validation sample, with relations. valid_x_d_c[0] is the features.
        :param valid_y: labels of validation sample/
        :param test_x_d_c: features of test sample. with relations. test_x_d_c[0] is the features.
        :param test_y: labels of test sample.
        :return: nothing.
        """
        self.valid_x_d_c = valid_x_d_c
        self.valid_y = valid_y
        self.test_x_d_c = test_x_d_c
        self.test_y = test_y
        self.save_result = True
        if self.test_x_d_c is not None:
            self.do_test = True

    def compute_result(self, x_d_c, y_true, sample_type):
        """
        invoked on epoch end.
        :param x_d_c: tensor of all features of all examples in certain set.(valid or test)
        :param y_true: correct labels of all examples. in the same set.
        :param sample_type: the type of sample to be analyzed: valid or test.
        :return: auc: area under curve. no idea what this does.
        :return: f1: combines precision and recall somehow.
        :return: pre: precision. out of the times the label dog was predicted, how many were actually dogs.
        :return: rec: recall. how many dogs out of all dogs were found.
        """
        valid_x = x_d_c[0]
        batch_size = valid_x.shape[0]
        y_pred = self.model.predict(x_d_c, batch_size=batch_size)
        logging.debug("computing result for : %s. the shape of y_true is : %s. the shape of y_pred is: %s \n" %
                      (sample_type, y_true.shape, y_pred.shape))
        if 'software' in self.task:
            y_pred = 1.0 - y_pred[:, 0]
            nonzero_ids = numpy.nonzero(y_true)[0]
            y_true[nonzero_ids] = 1

        if numpy.max(y_true) == 1:
            fp, tp, thresholds = metrics.roc_curve(y_true, y_pred)
            auc = metrics.auc(fp, tp)
            y_pred = numpy.round(y_pred)
            average = 'binary'
        else:
            y_pred = numpy.argmax(y_pred, axis=1)
            average = 'micro'
            auc = metrics.f1_score(y_true, y_pred, average='macro')

        if numpy.isnan(y_pred).any():
            return 0.0, 0.0, 0.0, 0.0

        # metric can be 'f1_binary', 'f1_micro', 'f1_macro' (for multi-classes)
        call = {'f1': metrics.f1_score,
                'recall': metrics.recall_score,
                'precision': metrics.precision_score}

        pre = call['precision'](y_true, y_pred, average=average)
        rec = call['recall'](y_true, y_pred, average=average)
        f1 = call['f1'](y_true, y_pred, average=average)
        accuracy = metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
        err = metrics.zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None)
        return accuracy, err, auc, f1, pre, rec

    def on_epoch_begin(self, epoch, logs=None):
        logging.debug("epoch starts")

    def on_epoch_end(self, epoch, logs={}):
        """

        :param epoch: the index of the current epoch. probably calculated automaticly.
        :param logs: ?
        :return: nothing.
        :operations: updates log files with results of current parameters on epoch, and updates best result paramters
                    for furure usage in case of neeed.
        """
        st = time.time()
        if not self.save_result:
            et = time.time()
            logging.debug("on_epoch_end, SaveResultCallback:  time: %f. " % (et - st))
            return
        self.n_epoch += 1
        self.save_result = False
        v_acc, v_err, v_auc, v_f1, v_pre, v_rec = self.compute_result(self.valid_x_d_c, self.valid_y,
                                                                      sample_type='valid')

        f = open(self.fileResult, 'a')
        f.write('e#: %d\n' % self.n_epoch)
        f.write('\tvalidation:\n')
        f.write('\t\t%.4f\t%.4f\t|\t%.4f\t%.4f\t%.4f\t%.4f\n' % (logs['loss'], logs['val_loss'], v_auc, v_f1, v_pre,
                                                                 v_rec))
        f.write('\t\t%.4f\t%.4f\n' % (v_acc, v_err))
        if self.do_test:
            t_acc, t_err, t_auc, t_f1, t_pre, t_rec = self.compute_result(self.test_x_d_c, self.test_y,
                                                                          sample_type='test')
            f.write('\ttest:\n')
            f.write('\t\t%.4f\t%.4f\t%.4f\t%.4f\n' % (t_auc, t_f1, t_pre, t_rec))
            f.write('\t\t%.4f\t%.4f\n' % (t_acc, t_err))

        if v_f1 > self.bestResult:
            self.bestResult = v_f1
            self.bestEpoch = self.n_epoch
            # this actually saves weights of net to file. note it is decided based on validation only.
            # no decistions are made based on test sample.
            self.model.save_weights(self.fileParams, overwrite=True)
            self.wait = 0
        f.write('best epoch so far, by v_f1: %d\n' % self.bestEpoch)
        f.close()

        if v_f1 < self.bestResult:
            self.wait += 1
            if self.wait == self.minPatience:
                self.wait = 0
                self.minPatience += 5

                lr = K.get_value(self.model.optimizer.lr) / 2.0
                K.set_value(self.model.optimizer.lr, lr)
                print ('New learning rate: %.4f', K.get_value(self.model.optimizer.lr))
                if self.minPatience > self.maxPatience:
                    self.model.stop_training = True
        et = time.time()
        logging.debug("on_epoch_end, SaveResultCallback:  time: %f. " % (et - st))




class NanStopping(Callback):
    def __init__(self):
        super(NanStopping, self).__init__()
        self.n_epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        self.n_epoch += 1
        # logging.debug("NanStopping.on_epoch_end was invoked." + str(self.n_epoch))
        for k in logs.values():
            if numpy.isnan(k):
                self.model.stop_training = True


# def graph_loss(y_true, y_pred):
#     ids = tensor.nonzero(y_true + 1)[0]
#     y_true = y_true[ids]
#     y_pred = y_pred[ids]
#     return tensor.mean(tensor.nnet.binary_crossentropy(y_pred, y_true))
#

# def multi_sparse_graph_loss(y_true, y_pred):
#     ids = tensor.nonzero(y_true[:,0] + 1)[0]
#     y_true = y_true[ids]
#     y_pred = y_pred[ids]
#
#     return tensor.mean(objectives.sparse_categorical_crossentropy(y_true, y_pred))


class Evaluator(Callback):
    def __init__(self, test_gen, file_result='', file_params='', min_patience=5, max_patience=20):
        super(Evaluator, self).__init__()  # call constructor of Callback Class.
        logging.info("Evaluator: constructor: started.")
        self.test_gen = test_gen
        self.log_file_path = file_result
        self.param_file_path = file_params

        self.bestResult = 0.0
        self.bestEpoch = 0
        # wait to divide the learning rate. if reach the maxPatience -> stop learning
        self.wait = 0
        self.minPatience = min_patience
        self.maxPatience = max_patience
        logging.info("Evaluator: constructor: ended.")

    def set_gen(self, new_gen):
        self.test_gen = new_gen

    def on_epoch_begin(self, epoch, logs=None):
        logging.info("Evaluator: on_epoch_begin: started.")
        logging.info("Evaluator: on_epoch_begin: ended.")

    def on_epoch_end(self, epoch, logs=None):
        logging.info("Evaluator: on_epoch_ended: started.")
        test_batch_generator = self.test_gen.gen
        test_size = self.test_gen.n_samples
        y_pred = self.model.predict_generator(test_batch_generator, test_size,
                                              max_q_size=10, nb_worker=1, pickle_safe=False)
        y_true = self.test_gen.get_ytrue()
        logging.debug("computing results. the shape of y_true is : %s. the shape of y_pred is: %s" %
                      (y_true.shape, y_pred.shape))
        if numpy.max(y_true) == 1:
            fp, tp, thresholds = metrics.roc_curve(y_true, y_pred)
            auc = metrics.auc(fp, tp)
            y_pred = numpy.round(y_pred)
            average = 'binary'
        else:
            y_pred = numpy.argmax(y_pred, axis=1)
            average = 'micro'
            auc = metrics.f1_score(y_true, y_pred, average='macro')

        if numpy.isnan(y_pred).any():
            return 0.0, 0.0, 0.0, 0.0

        # metric can be 'f1_binary', 'f1_micro', 'f1_macro' (for multi-classes)
        call = {'f1': metrics.f1_score,
                'recall': metrics.recall_score,
                'precision': metrics.precision_score}

        pre = call['precision'](y_true, y_pred, average=average)
        rec = call['recall'](y_true, y_pred, average=average)
        f1 = call['f1'](y_true, y_pred, average=average)
        accuracy = metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
        err = metrics.zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None)

        logging.info("Evaluator: on_epoch_end: done testing. acc. %f. err: %f." % (accuracy, err))
        f = open(self.log_file_path, 'a')
        f.write('\t\t%.4f\t%.4f\t%.4f\t%.4f\n' % (auc, f1, pre, rec))
        f.write('\t\t%.4f\t%.4f\n' % (accuracy, err))
        f.close()

        if f1 > self.bestResult:
            logging.debug("Evaluator: on_epoch_end: best result imporoved! saving weights!")
            self.bestResult = f1
            # this actually saves weights of net to file. note it is decided based on validation only.
            # no decistions are made based on test sample.
            self.model.save_weights(self.param_file_path, overwrite=True)
            self.wait = 0
        f.close()
        if f1 < self.bestResult:
            logging.debug("Evaluator: on_epoch_end: best result was not achieved this time.")
            self.wait += 1
            if self.wait == self.minPatience:
                self.wait = 0
                self.minPatience += 5
                lr = K.get_value(self.model.optimizer.lr) / 2.0
                K.set_value(self.model.optimizer.lr, lr)
                print ('New learning rate: %.4f', K.get_value(self.model.optimizer.lr))
                if self.minPatience > self.maxPatience:
                    self.model.stop_training = True
        logging.info("Evaluator: on_epoch_ended: ended.")
