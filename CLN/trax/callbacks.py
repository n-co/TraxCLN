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
        test_batch_generator = self.test_gen.data_generator()
        test_size = self.test_gen.n_samples
        metrics_ans = self.model.evaluate_generator(generator=test_batch_generator, val_samples=test_size,
                                              max_q_size=10, nb_worker=1, pickle_safe=False)
        print(self.model.metrics_names)
        print(metrics_ans)

        f1 = metrics_ans[2]

        f = open(self.log_file_path, 'a')
        f.write("%s\n"%str(metrics_ans))
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
