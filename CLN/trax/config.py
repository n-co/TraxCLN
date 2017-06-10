import sys
sys.path.insert(0, '../../')
import logger
import logging
logger.updateLogger()
logging.getLogger().setLevel(logging.INFO)
run_mode = 'run'
logging.debug("Logger is up and running!")

product_height = 100
product_width = 300
product_channels = 3
product_shape = (product_height, product_width, product_channels)
product_hw = (product_height, product_width)


data_sets_dir = "/vildata/rawdata/Trax/proj_nir_noam/TraxInputData/"

models_path = "/vildata/derdata/Trax/proj_nir_noam/models/"
best_models_path = "/vildata/derdata/Trax/proj_nir_noam/best_models/"
logs_path = "/vildata/derdata/Trax/proj_nir_noam/logs/"

number_of_epochs = 100
learning_rate = 0.001
dropout_ratio = 0.5
dropout_ration_cnn = 0.2
bias_factor = 0.1
