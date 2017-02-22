#from config import *
import config as conf
import json
import cv2 as ocv
import numpy as np


class Probe:
    def __init__(self, probe_id):
        self.id = probe_id
        self.path = conf.probes_dir + probe_id + ".jpg"
        self.features = ocv.imread(self.path)

        self.rights = None
        self.lefts = None
        self.products = np.array([])

    def set_rights(self, rights):
        self.rights = rights

    def set_lefts(self, lefts):
        self.lefts = lefts


class Product:
    def __init__(self, id, brand_code, sample_type, brand_label, form_factor_label, mask, object_label, patch_id, patch_url,
                 probe_id, product_label, voting_confidence, probe_obj ):
        self.id = int(id)
        self.brand_code = brand_code
        self.sample_type = sample_type
        self.brand_label = int(brand_label)
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
        self.voting_confidence = json.loads(voting_confidence)  # TODO: convert to json
        self.probe_obj = probe_obj

        self.path = conf.products_dir + patch_url + ".jpg"
        self.features = populate_features(self)
        # self.relations = np.empty([2, 0], dtype=type(np.ndarray))
        self.relations = [[],[]]        # print self.relations
        self.index_in_probe = -1

    def build_relations(self):
        probe_obj = self.probe_obj
        rights_matrix = probe_obj.rights
        lefts_matrix = probe_obj.lefts
        products = probe_obj.products
        my_index = self.index_in_probe

        for j in range(0, len(products)):
            if rights_matrix[my_index][j] == 1:
                self.relations[conf.rel_right].append(products[j].id)
                # self.relations[conf.rel_right] = np.append(self.relations[conf.rel_right], products[j].id)

        for j in range(0, len(products)):
            if lefts_matrix[my_index][j] == 1:
                self.relations[conf.rel_left].append(products[j].id)
                # self.relations[conf.rel_left] = np.append(self.relations[conf.rel_left], products[j].id)


def populate_features(self):
    probe_path = conf.probes_dir + self.probe_id + ".jpg"
    x1 = self.mask["x1"]
    x2 = self.mask["x2"]
    y1 = self.mask["y1"]
    y2 = self.mask["y2"]
    # img = ocv.imread(probe_path)
    img = self.probe_obj.features
    cropped = img[y1:y2, x1:x2]
    cropped = ocv.resize(cropped, conf.product_hw)  # resize image

    return cropped




