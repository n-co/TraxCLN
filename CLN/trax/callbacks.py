from keras.callbacks import *
import numpy as np
import logging
import datetime as dt


class NanStopping(Callback):
    """
    inherits from Callback. will be used on epoch end to check if we should stop running.
    """

    def __init__(self):
        logging.debug("NanStopping: constructor: started.")
        super(NanStopping, self).__init__()
        self.n_epoch = -1
        logging.debug("NanStopping: constructor: ended.")

    def on_epoch_end(self, epoch, logs={}):
        logging.debug("NanStopping: on_epoch_end: started.")
        self.n_epoch += 1
        for k in logs.values():
            if np.isnan(k):
                self.model.stop_training = True
        logging.debug("NanStopping: on_epoch_end: ended.")


class Evaluator(Callback):
    """
    inherits from Callback. will be used on epoch end to evaluate the performance of the network.
    """

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

    def on_epoch_end(self, epoch, logs=None):
        """
        evaluates model on all the generators it has: validation and test.
        logs the results.
        updates best models and learning rate if needed, according to information from validation set.
        :param epoch:
        :param logs:
        :return:
        """
        logging.info("Evaluator: on_epoch_ended: started.")
        self.n_epoch += 1

        # reset all available generators.
        gens = []
        meta_gen = [self.train_gen, self.valid_gen, self.test_gen]
        for x in meta_gen:
            if x is not None:
                gens.append([x.data_generator(), x.n_samples])

        # for every available generator, evalute model on it.
        metrics_ans = []
        for i in range(len(gens)):
            res = self.model.evaluate_generator(generator=gens[i][0], val_samples=gens[i][1],
                                                max_q_size=10, nb_worker=1, pickle_safe=False)
            metrics_ans.append(res)

        f = open(self.log_file_path, 'a')
        now = dt.datetime.now().replace(microsecond=0)
        f.write("\n%s %4s: " % (now, self.n_epoch))
        # for every available generator, log its results.
        for i in range(len(gens)):
            f.write('       ')
            for ans in metrics_ans[i]:
                # 18 digits after decimal point, and room for at least 22 chars ==> '0.234567890123456789  '
                f.write('%-22.18f' % ans)
        f.close()

        # compare result to best result so far
        f1 = metrics_ans[0][2]  # 0: valid. 2: f1.
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
                logging.info('New learning rate: %.4f' % K.get_value(self.model.optimizer.lr))
                if self.minPatience > self.maxPatience:
                    self.model.stop_training = True
        logging.debug("Evaluator: on_epoch_ended: ended.")
