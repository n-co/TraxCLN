import sys
sys.path.insert(0, '../')
import logger
import logging
logger.updateLogger()
logging.getLogger().setLevel(logging.INFO)
run_mode = 'run'
logging.debug("Logger is up and running!")

# parameters for product sizes
product_width = 100
product_height = 300
product_channels = 3
product_shape = (product_height, product_width, product_channels)
product_wh = (product_width, product_height)
product_size = product_wh[0] * product_wh[1] * product_channels
size_string = "_" + str(product_width) + "_" + str(product_height)

raw_data_dir = "/vildata/rawdata/Trax/proj_nir_noam/TraxInputData"
csv_path = raw_data_dir + "/data_full.csv"
probes_dir = raw_data_dir + "/Probes/"
framed_probes_dir = raw_data_dir + "/FramedProbes/"
products_dir = raw_data_dir + "/Products" + size_string + "/"
pickle_path = raw_data_dir + "/trax" + size_string + "_" + csv_path.strip(raw_data_dir).strip(".csv").strip("/") + ".pkl"


pad_type = 'black'  # 'black' or 'noise'

csv_length = -1

probes_for_train = 12000
probes_for_valid = 2000
probes_for_test = 2000

eps = 200  # max distance (in pixels) between adjacent products


def sort_key(product):
    # sort_key of a product in a "shelf"
    return (product.mask["x1"] + product.mask["x2"]) / 2


def dist(pr1, pr2):
    # distance function between two products
    y1 = pr1.mask["y2"]
    y2 = pr2.mask["y2"]
    return abs(y1 - y2)

# enums
rel_left = 0
rel_right = 1

train_counter = 0
valid_counter = 0
test_counter = 0
last_sample_type = 'train'
prev_probe_id = None


def get_sample_type(probe_id):
    global train_counter, valid_counter, test_counter, last_sample_type, prev_probe_id
    if probe_id == prev_probe_id:
        return last_sample_type
    if train_counter < probes_for_train:
        train_counter += 1
        last_sample_type = 'train'
        prev_probe_id = probe_id
        return 'train'
    elif valid_counter < probes_for_valid:
        valid_counter +=1
        last_sample_type = 'valid'
        prev_probe_id = probe_id
        return 'valid'
    else:
        test_counter +=1
        last_sample_type = 'test'
        prev_probe_id = probe_id
        return 'test'


def should_i_stop(probe_id):
    global train_counter, valid_counter, test_counter, last_sample_type, prev_probe_id
    return test_counter > probes_for_test



logging.debug("Loaded configuration file.")
