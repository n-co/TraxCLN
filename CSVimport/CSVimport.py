import csv
import json
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import cv2 as ocv
import itertools

np.set_printoptions(threshold='nan')

class Picture:
    def __init__(self, rights, lefts):
        # type: (object, object) -> object
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
    my_list = {}
    my_pics = {}
    with open('../CSV/data.csv', 'r') as f:
        lines = itertools.islice(f, 1, None)
        reader = csv.reader(lines)
        for row in reader:
            ex = Example(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10])
            my_list[ex.probe_id] = []
            my_pics[ex.probe_id] = {}
        f.seek(0)
        lines = itertools.islice(f, 1, None)
        reader = csv.reader(lines)
        for row in reader:
            ex = Example(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10])
            my_list[ex.probe_id].append(ex)

    return {"list": my_list, "pics": my_pics}


tmp = import_data()
li = tmp["list"]
pics = tmp["pics"]
# print li[0].mask
# print li[2].voting_confidence
# print li[2].get_coordinates()
# print li[2].get_voting_confidence()

# print li[15].get_image()
ocv.waitKey(0)  # show plots
# for ex in li:
#     ex.write_image()
num_of_pics = len(li)
for probe_id in li:
    curr = li[probe_id]
    n = len(curr)
    # print probe_id
    # print n
    Rights = np.zeros((n, n))
    for i in range(0, n):
        meee = curr[i]
        for j in range(0, n):
            otherrrr = curr[j]
            Rights[i][j] = is_on_right(meee, otherrrr)
    pics[probe_id] = Picture(Rights, np.transpose(Rights))
print pics["10497567"].rights
print "=========================================================="
print pics["10497567"].lefts

# print Right
# print Right.shape
# print np.sum(Right)
# print is_on_right(li[30], li[31])
