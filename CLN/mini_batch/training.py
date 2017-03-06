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

# Load data
logging.info("Process args and Load data - Started.")
args = arg_passing(sys.argv)
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
modelType = args['-model']
n_layers = args['-nlayers']
dim = args['-dim']
shared = args['-shared']
saving = args['-saving']
nmean = args['-nmean']
yidx = args['-y']
batch_size = int(args['-batch'])
if 'dr' in args['-reg']:
        dropout = True
else:
    dropout = False
feats, labels, rel_list, rel_mask, train_ids, valid_ids, test_ids = load_data(dataset)
labels = labels.astype('int64')
if task == 'movie':
    labels = labels[:, yidx]

# the number of classes is max+1 since the first class is 0.
# in binary classification, this parameter is probably not used.
n_classes = numpy.max(labels)
if n_classes > 1:
    n_classes += 1
    loss = sparse_categorical_crossentropy
else:
    loss = binary_crossentropy

# define the optimizer for weights finding.
all_optimizers = {'RMS': RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8),
       'Adam': Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)}
# opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# opt = Adagrad(learning_rate=0.01, epsilon=1e-8)
# opt = Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-8)
# opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# opt = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8
selected_optimizer = all_optimizers[args['-opt']]
logging.info("Process args and Load data - Ended.")

# Build Model.
logging.info("Build Model - Started")

if modelType == 'Highway':
    model = create_highway(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1],
                           n_rel=rel_list.shape[-2], n_neigh=rel_list.shape[-1],
                           n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
elif modelType == 'Dense':
    model = create_dense(n_layers=n_layers, hidden_dim=dim, input_dim=feats.shape[-1], n_rel=rel_list.shape[-2],
                         n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)
elif modelType == 'HCNN':
    model = create_hcnn(n_layers=n_layers, hidden_dim=dim, input_shape=feats[0].shape, n_rel=rel_list.shape[-2],
                        n_neigh=rel_list.shape[-1], n_classes=n_classes, shared=shared, nmean=nmean, dropout=dropout)

model.summary()
model.compile(optimizer=selected_optimizer, loss=loss)

logging.info("Build Model - Ended.")


logging.info("Last Configurations before launching network - Started.")
# Log information so far.

# Prints the model, in a json format, to the desired path.
json_string = model.to_json()
fModel = models_path + saving + '.json'
f = open(fModel, 'w')
f.write(json_string)
f.close()

# Define path for saving results.
fParams = best_models_path + saving + '.hdf5'

# Create a log.
fResult = logs_path + saving + '.txt'
f = open(fResult, 'w')
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


hidden_data = HiddenSaving(n_samples=feats.shape[0], n_layers=n_layers, h_dim=dim, rel=rel_list, rel_mask=rel_mask)

# created a variable containing a generator, that upon request generates a list of, possibly random, ids to select from
# TRAIN SET.
mini_batch_ids_generator = MiniBatchIds(len(train_ids), batch_size=batch_size)

# Calculate how many batches are there.
n_batchs = len(train_ids) // batch_size
if len(train_ids) % batch_size > 0:
    n_batchs += 1

# Exctract features and labels of validation and sample tests.
valid_x, valid_y = feats[valid_ids], labels[valid_ids]
test_x, test_y = feats[test_ids], labels[test_ids]

performence_evaluator = SaveResult(task=task, file_result=fResult, file_params=fParams)
callbacks = [performence_evaluator, NanStopping()]
hidd_funcs = get_hidden_funcs(model, n_layers)
logging.info("Last Configurations before launching network - Ended.")

logging.info("CLN - Started.")
for epoch in xrange(number_of_epochs):  # train the network a few times to get more accurate results.
    start = time.time()
    for i in xrange(n_batchs):  # go over all batches. all of training data will be used.
        # Extract features and labels of TRAIN set.
        mini_batch_ids = train_ids[mini_batch_ids_generator.get_mini_batch_ids(i)]
        train_x = feats[mini_batch_ids]
        train_y = labels[mini_batch_ids]

        # get hidden context for train set.
        train_context = hidden_data.get_context(mini_batch_ids)

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
                new_hiddens = get_hiddens(model, x, ct, hidd_funcs)
                hidden_data.update_hidden(ids, new_hiddens)

            # define validation data.
            valid_data = [[valid_x] + valid_context, valid_y]
        else:
            valid_data = None

        his = model.fit([train_x] + train_context, numpy.expand_dims(train_y, -1),
                        validation_data=valid_data, verbose=0,
                        nb_epoch=1, batch_size=train_x.shape[0], shuffle=False,
                        callbacks=callbacks)
                        # nb_epoch: how many times to iterate. this loop imelments iterations by itself.

        # update hidden data for train set.
        new_train_hiddens = get_hiddens(model, train_x, train_context, hidd_funcs)
        hidden_data.update_hidden(mini_batch_ids, new_train_hiddens)
    end = time.time()
    logging.info('epoch %d, runtime: %.1f' % (epoch, end - start))
    if model.stop_training:
        break
logging.info("CLN - Ended.")