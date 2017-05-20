from prepare_data import *
class SampleGenerator():

    # global batch_size,task

    def __init__(self, sample_index, sample_name):
        global p, l, rl, rm, b
        global batch_size,task
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
            logging.debug("generating batch for sample: " + self.name + " " + str(self.curr_batch) + "  " + str(self.max_batches) + "   " + str(ids))
            if ids is not []:
                sx, sy, srl, srm = self.prepare_batch(ids, p[self.si], l[self.si], rl[self.si], rm[self.si])
                # stop_and_read('debug')
            self.curr_batch += 1
            return [[sx, srl, srm], sy]

    def batches_generator(self):
        while True:
            ids = self.train_ids_gen.get_mini_batch_ids(self.curr_batch)
            logging.debug("generating batch for sample: " + self.name + " " + str(self.curr_batch) + "  " + str(self.max_batches) + "   " + str(ids))
            if ids != np.array([]):
                sx, sy, srl, srm = self.prepare_batch(ids, p[self.si], l[self.si], rl[self.si], rm[self.si])
                # stop_and_read('debug')
            self.curr_batch += 1
            if ids != np.array([]):
                yield [sx, srl, srm], sy
            else:
                logging.error("overhead!")
                self.curr_batch = 0

    def build_gen(self):
        self.curr_batch =0
        self.gen = self.batches_generator()