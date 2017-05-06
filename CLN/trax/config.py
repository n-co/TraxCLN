import sys
import logging
sys.path.insert(0, '../../')
import logger
from tools import *
logging.getLogger().setLevel(logging.DEBUG)
run_mode = 'run'
logging.debug("Logger is up and running - mini batch training.")

product_height = 100
product_width = 300
product_channels = 3
product_shape = (product_height, product_width, product_channels)
product_hw = (product_height, product_width)

models_path = "models/"
best_models_path = "bestModels/"
logs_path = "log/"
data_sets_dir = "/vildata/rawdata/Trax/proj_nir_noam/TraxInputData/"
number_of_epochs = 100  # TODO: 1000
learning_rate = 0.001
dropout_ratio = 0.5
dropout_ration_cnn = 0.2
bias_factor = 0.1
