import sys
import logging
import cv2 as ocv
import numpy as np
sys.path.insert(0, '../../')
import logger
logging.getLogger().setLevel(logging.DEBUG)
logging.info("Logger is up and running - mini batch training.")

product_height =32
product_width = 32
product_channels = 3
product_shape = (product_height, product_width, product_channels)
product_hw = (product_height, product_width)

models_path = "models/"
best_models_path = "bestModels/"
logs_path = "log/"
data_sets_dir = "../../TraxOutputData/data/"
number_of_epochs = 10  # TODO: 1000
learning_rate = 0.001
dropout_ratio = 0.5
dropout_ration_cnn = 0.2
bias_factor = 0.1
