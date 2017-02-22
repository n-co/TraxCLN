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

csv_path = "../TraxInputData/data.csv"
probes_dir = "../TraxInputData/Probes/"
products_dir = "../TraxInputData/Products/"
pickle_path = "../TraxOutputData/data/trax.pkl"

product_hw = (100,100)

csv_length = -1

gap_ratio_x = 0.5
gap_ratio_y = 0.5

# enums
rel_left = 0
rel_right = 1


# np.set_printoptions(threshold='nan')  # show big matrix

#commands
# f.seek(0)
#show_product_image("noam", probes, '10521992', 2)
