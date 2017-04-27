from classes import *
import csv
import json
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import cv2 as ocv
import itertools
import numpy as np
import glob
import cPickle
import gzip
import pickle
import math
import sys
sys.path.insert(0, '../')
import logger
import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.info("Logger is up and running!")

product_width = 200
product_height = 600
product_channels = 3
product_shape = (product_height, product_width, product_channels)
product_wh = (product_width, product_height)
product_size = product_wh[0] * product_wh[1] * product_channels

size_string = "_" + str(product_width) + "_" + str(product_height)

raw_data_dir = "/vildata/rawdata/Trax/proj_nir_noam/TraxInputData"
csv_path = raw_data_dir + "/data.csv"
probes_dir = raw_data_dir + "/Probes/"
framed_probes_dir = raw_data_dir + "/FramedProbes/"
products_dir = raw_data_dir + "/Products" + size_string + "/"
pickle_path = raw_data_dir + "/trax" + size_string + ".pkl"


pad_type = 'black'  # 'black' or 'noise'

csv_length = -1
csv_length_limit = 50615  # 290000
first_valid_index = 50013  # 289400  # TODO: make sure these values start at the begining of a probe.
first_test_index = 50322  # 289700

probes_for_train = 2000  # 40000  # products
probes_for_valid = 500  # 10000  # products
probes_for_test = 500  # 10000  # products

eps = 200  # max distance (in pixels) between adjacent products


def sort_key(product):
    # sort_key of a product in a "shelf"
    return (product.mask["x1"] + product.mask["x2"]) / 2


def dist(pr1, pr2):
    # distance function between two products
    y1 = pr1.mask["y2"]
    y2 = pr2.mask["y2"]
    return abs(y1 - y2)

train_counter = 0
valid_counter = 0
test_counter = 0
last_sample_type = 'train'
prev_probe_id = None


def get_sample_type(probe_id):
    global train_counter,valid_counter, test_counter,last_sample_type,prev_probe_id
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

#
# def get_sample_type(index):
#     if 0 <= index <= first_valid_index - 1:
#         sample_type = 'train'
#     elif first_valid_index <= index <= first_test_index - 1:
#         sample_type = 'valid'
#     else:
#         sample_type = 'test'
#     return sample_type


# def should_i_stop(index):
#     if csv_length_limit == -1:
#         return False
#     else:
#         return index > csv_length_limit

def should_i_stop(probe_id):
    global train_counter, valid_counter, test_counter, last_sample_type, prev_probe_id
    return test_counter > probes_for_test

# enums
rel_left = 0
rel_right = 1

logging.info("Loaded configuration file.")

# commands
# f.seek(0)
# show_product_image("noam", probes, '10521992', 2)
# img = ocv.imread(probe_path)
# feats, labels, rel_list, train_ids, valid_ids, test_ids = load_gzip_file(pickle_path + ".gz")
