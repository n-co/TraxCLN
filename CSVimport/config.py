from Classes import *
import csv
import json
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import cv2 as ocv
import itertools
import numpy as np
import glob


csv_path = "../TraxInputData/data.csv"
probes_dir = "../TraxInputData/Probes/"
products_dir = "../TraxInputData/Products/"

gap_ratio_x = 0.5
gap_ratio_y = 0.5


# np.set_printoptions(threshold='nan')  # show big matrix

#commands
# f.seek(0)