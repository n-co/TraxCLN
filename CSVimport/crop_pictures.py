import csv
import json
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import cv2 as ocv
import itertools

csv_path = "../TraxInputData/data.csv"
probes_dir = "../TraxInputData/Probes/"
bottles_dir = "../TraxInputData/Bottles/"


class Bottle:
    def __init__(self, id, brand_code, brand_label, form_factor_label, mask, object_label, patch_id, patch_url,
                 probe_id, product_label, voting_confidence):
        self.id = id
        self.brand_code = brand_code
        self.brand_label = brand_label
        self.form_factor_label = form_factor_label
        mask = mask.replace('\'', '\"')  # fix json string
        self.mask = json.loads(mask)
        self.object_label = object_label
        self.patch_id = patch_id
        self.patch_url = patch_url
        self.probe_id = probe_id
        self.product_label = product_label
        voting_confidence = voting_confidence.replace('\'', '\"')
        voting_confidence = voting_confidence.replace('u', '')
        self.voting_confidence = voting_confidence  # TODO: convert to json

    def get_cropped_image(self):
        probe_path = probes_dir + self.probe_id + ".jpg"
        x1 = self.mask["x1"]
        x2 = self.mask["x2"]
        y1 = self.mask["y1"]
        y2 = self.mask["y2"]
        img = ocv.imread(probe_path)
        cropped = img[y1:y2, x1:x2]
        return cropped


def import_data():
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        for row in reader:
            bottle = Bottle(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10])
            cropped = bottle.get_cropped_image()
            ocv.imwrite(bottles_dir + bottle.patch_url, cropped)

import_data()
