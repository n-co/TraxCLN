import sys
import logging
import cv2 as ocv
import numpy as np
import time
sys.path.insert(0, '../../')
import logger
from tools import *
logging.getLogger().setLevel(logging.DEBUG)
run_mode = 'run'
logging.info("Logger is up and running - mini batch training.")

product_height = 200
product_width = 600
product_channels = 3
product_shape = (product_height, product_width, product_channels)
product_hw = (product_height, product_width)

models_path = "models/"
best_models_path = "bestModels/"
logs_path = "log/"
data_sets_dir = "/vildata/rawdata/Trax/proj_nir_noam/TraxInputData/"
number_of_epochs = 10  # TODO: 1000
learning_rate = 0.001
dropout_ratio = 0.5
dropout_ration_cnn = 0.2
bias_factor = 0.1
