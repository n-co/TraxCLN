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

raw_data_dir = "/vildata/rawdata/Trax/proj_nir_noam/TraxInputData"
csv_path = raw_data_dir + "/data.csv"
probes_dir = raw_data_dir + "/Probes/"
framed_probes_dir = raw_data_dir + "FramedProbes/"
products_dir = raw_data_dir + "/Products/"
pickle_path = raw_data_dir + "/trax.pkl"


product_height = 200
product_width = 600
product_channels = 3
product_shape = (product_height, product_width, product_channels)
product_hw = (product_height, product_width)

product_size = product_hw[0]*product_hw[1] * product_channels

csv_length = -1
csv_length_limit = 50600 #290000
first_valid_index = 50000 #289400
first_test_index = 50300 #289700

# gap_ratio_x = 0.5
# gap_ratio_y = 0.5

eps = 200  # max distance between adjacent products


def sort_key(product):
    # sort_key of a product in a "shelf"
    return (product.mask["x1"] + product.mask["x2"]) / 2


def dist(pr1, pr2):
    # distance function between two products
    # x1 = (pr1.mask["x1"] + pr1.mask["x2"]) / 2
    y1 = pr1.mask["y2"]
    # x2 = (pr2.mask["x1"] + pr2.mask["x2"]) / 2
    y2 = pr2.mask["y2"]
    return abs(y1 - y2)

def get_sample_type(index):
    if index >= 0 and index <= first_valid_index - 1:
        sample_type = 'train'
    elif index >= first_valid_index and index <= first_test_index - 1:
        sample_type = 'valid'
    else:
        sample_type = 'test'
    return sample_type


def should_i_stop(index):
    if csv_length_limit == -1:
        return False
    else:
        return index > csv_length_limit

# enums
rel_left = 0
rel_right = 1

logging.info("Loaded configuration file.")

# commands
# f.seek(0)
# show_product_image("noam", probes, '10521992', 2)
# img = ocv.imread(probe_path)
# feats, labels, rel_list, train_ids, valid_ids, test_ids = load_gzip_file(pickle_path + ".gz")
