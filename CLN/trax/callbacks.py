from config import *
import numpy
import datetime as dt
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
        # logging.info("NanStopping.on_epoch_end was invoked." + str(self.n_epoch))
        for k in logs.values():
            if numpy.isnan(k):
                self.model.stop_training = True


class Evaluator(Callback):
    def __init__(self, file_result='', file_params='', min_patience=5, max_patience=20):
        super(Evaluator, self).__init__()  # call constructor of Callback Class.
        logging.debug("Evaluator: constructor: started.")
        self.train_gen = None
        self.valid_gen = None
        self.test_gen = None
        self.log_file_path = file_result
        self.param_file_path = file_params
        self.n_epoch = 0
        self.bestResult = 0.0
        self.bestEpoch = 0
        # wait to divide the learning rate. if reach the maxPatience -> stop learning
        self.wait = 0
        self.minPatience = min_patience
        self.maxPatience = max_patience
        logging.debug("Evaluator: constructor: ended.")

    def on_epoch_begin(self, epoch, logs=None):
        logging.debug("Evaluator: on_epoch_begin: started.")
        logging.debug("Evaluator: on_epoch_begin: ended.")

    def on_epoch_end(self, epoch, logs=None):
        logging.debug("Evaluator: on_epoch_ended: started.")
        self.n_epoch +=1
        gens = []
        if self.train_gen is not None:
            gens.append([self.train_gen.data_generator(),self.train_gen.n_samples])
        if self.valid_gen is not None:
            gens.append([self.valid_gen.data_generator(),self.valid_gen.n_samples])
        if self.test_gen is not None:
            gens.append([self.test_gen.data_generator(),self.test_gen.n_samples])
        metrics_ans = []
        for i in range(len(gens)):
            res = self.model.evaluate_generator(generator=gens[i][0], val_samples=gens[i][1],
                                                        max_q_size=10, nb_worker=1, pickle_safe=False)
            metrics_ans.append(res)

        f = open(self.log_file_path, 'a')
        now = dt.datetime.now().replace(microsecond=0)
        f.write("\n%s %4s:" % (now, self.n_epoch))
        for i in range(len(gens)):
            f.write("%s--" % metrics_ans[i])
        f.close()

        f1 = metrics_ans[0][2]  # TODO: make sure we learn from f1 of valid
        if f1 > self.bestResult:
            logging.info("Evaluator: on_epoch_end: best result imporoved! saving weights!")
            self.bestResult = f1
            # this actually saves weights of net to file. note it is decided based on validation only.
            # no decistions are made based on test sample.
            self.model.save_weights(self.param_file_path, overwrite=True)
            self.wait = 0
        f.close()
        if f1 < self.bestResult:
            logging.info("Evaluator: on_epoch_end: best result was not achieved this time.")
            self.wait += 1
            if self.wait == self.minPatience:
                self.wait = 0
                self.minPatience += 5
                lr = K.get_value(self.model.optimizer.lr) / 2.0
                K.set_value(self.model.optimizer.lr, lr)
                print ('New learning rate: %.4f', K.get_value(self.model.optimizer.lr))
                if self.minPatience > self.maxPatience:
                    self.model.stop_training = True
        logging.debug("Evaluator: on_epoch_ended: ended.")
