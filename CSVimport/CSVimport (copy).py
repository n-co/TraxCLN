import csv
import json
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import cv2 as ocv
import itertools

class Picture:
    def __init__(self, rights, lefts):
        self.rights = rights
        self.lefts = lefts

    def set_rights(self, rights):
        self.rights = rights

    def set_lefts(self, lefts):
        self.lefts = lefts

class Example:
    # id = 0
    # brand_code = 0
    # brand_label = 0
    # form_factor_label = 0
    # mask = 0
    # object_label = 0
    # patch_id = 0
    # patch_url = 0
    # probe_id = 0
    # product_label = 0
    # voting_confidence = 0

    def __init__(self, id, brand_code, brand_label, form_factor_label, mask, object_label, patch_id, patch_url,
                 probe_id,
                 product_label, voting_confidence):
        self.id = id
        self.brand_code = brand_code
        self.brand_label = brand_label
        self.form_factor_label = form_factor_label
        mask = mask.replace('\'', '\"')
        self.mask = mask  # TODO: convert to json
        self.object_label = object_label
        self.patch_id = patch_id
        self.patch_url = patch_url
        self.probe_id = probe_id
        self.product_label = product_label
        voting_confidence = voting_confidence.replace('\'', '\"')
        voting_confidence = voting_confidence.replace('u', '')
        self.voting_confidence = voting_confidence  # TODO: convert to json

    def get_coordinates(self):
        js = json.loads(self.mask)
        return js

    def get_voting_confidence(self):
        js = json.loads(self.voting_confidence)
        return js

    def get_image(self):
        image_path = "../CSV/Probes/" + self.probe_id + ".jpg"
        coords = self.get_coordinates()
        x1 = coords["x1"]
        x2 = coords["x2"]
        y1 = coords["y1"]
        y2 = coords["y2"]
        img = ocv.imread(image_path)
        # ocv.imshow("original",img)
        cropped = img[y1:y2, x1:x2]
        # ocv.imshow("cropped", cropped)
        return cropped

    def write_image(self):
        cropped = self.get_image()
        ocv.imwrite("../CSV/bottles/" + self.patch_url, cropped)


def is_on_right(me, other):
    if me == other:
        return False
    my_coords = me.get_coordinates()
    other_coodrds = other.get_coordinates()
    my_width = my_coords["x2"]-my_coords["x1"]
    delta_x = np.abs(my_coords["x2"] - other_coodrds["x1"])
    delta_y = np.abs(my_coords["y2"] - other_coodrds["y2"])
    my_height = my_coords["y2"] - my_coords["y1"]
    if delta_x <= 0.5 * my_width and delta_y <= 0.5 * my_height:
        return True
    else:
        return False


def import_data():
    my_list = []
    with open('../CSV/data.csv', 'r') as f:
        lines = itertools.islice(f, 1, None)
        reader = csv.reader(lines)
        for row in reader:
            ex = Example(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10])
            if ex.probe_id == "9816481":  # TODO: remove this
                my_list.append(ex)
    return my_list


li = import_data()
# print li[0].mask
# print li[2].voting_confidence
# print li[2].get_coordinates()
# print li[2].get_voting_confidence()

# print li[15].get_image()
ocv.waitKey(0)  # show plots
# for ex in li:
#     ex.write_image()
n = len(li)
Right = np.zeros((n, n))
for i in range(0, n):
    me = li[i]
    for j in range(0, n):
        other = li[j]
        Right[i][j] = is_on_right(me, other)
        print Right[i][j]

# np.set_printoptions(threshold='nan')
print Right
print Right.shape
print np.sum(Right)
# print is_on_right(li[30], li[31])


arr = {}
arr['x'] = 5
print arr['x']