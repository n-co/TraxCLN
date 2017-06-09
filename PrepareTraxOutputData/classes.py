from config import *
import numpy as np
import cv2 as ocv
import json


class StringToIntConvertor:
    """
    this class is designed to be a simple hash function from strings to ints.
    """

    def __init__(self):
        """
        mapping: the dictionary to hold <key,val> pairs. key=string. val=int.
        counter: number of values in dictionary.
        """
        self.mapping = {}
        self.counter = 0

    def string_to_int(self, s):
        """

        :param s: recieves a string to be converted to an integer.
        :return: if string exists in dicitionary, it returns its representation. if not, it gives this strign a new integer representation,
        saves it in dictionary, and returns it.
        """

        ans = -1
        if s in self.mapping:
            ans = self.mapping[s]
        else:
            self.mapping[s] = self.counter
            ans = self.counter
            self.counter += 1
        return ans

#create different hashers to be used later on.
product_label_hasher = StringToIntConvertor()
probe_to_batch_hasher = StringToIntConvertor()
brand_label_hasher = StringToIntConvertor()


class Probe:
    """
    this class represents a probe - an image of products from a store.
    """
    def __init__(self, probe_id):
        """

        :param probe_id: the probe id.
        id: the probe id.
        path: path to the file in which probe image is located.
        shelves: a list of all the shelves in this probe.
        products: a list of all the products in this probes.
        rights: a list of all the right relations in this probe.
        lefts: a list of all the lft relations in this probe.
        """
        self.id = str(probe_id)
        self.path = str(probes_dir + probe_id + ".jpg")
        self.shelves = None
        self.products = np.array([])
        self.rights = None
        self.lefts = None

    def set_shelves(self, shelves):
        """
        sets the shelves list.
        :param shelves: shelves list.
        :return:
        """
        self.shelves = shelves

    def set_rights(self, rights):
        """
        sets the rights list.
        :param rights: rights list.
        :return:
        """
        self.rights = rights

    def set_lefts(self, lefts):
        """
        sets the lefts list.
        :param lefts: lefts list.
        :return:
        """
        self.lefts = lefts


    def build_relations(self):
        """
        goes over shelves, that is assumed to be sorted according to order of appearnce
        on probe, by the time this method is executed.
        the selves contains instances of products, and this methods updates their relations.
        :return:
        """
        for shelf in self.shelves:
            # work on left.
            for i in range(1, len(shelf)):
                product = shelf[i]
                neighbour = shelf[i-1]
                product.relations[rel_left].append(neighbour.id)
            # work on right.
            for i in range(0, len(shelf)-1):
                product = shelf[i]
                neighbour = shelf[i+1]
                product.relations[rel_right].append(neighbour.id)


class Product:
    """
    this class represents a product in a probe.
    1 probe contains roughly 20 products.
    a product is mostlry a bottle.
    """
    def __init__(self, id, brand_code, sample_type, brand_label, form_factor_label, mask, object_label, patch_id, patch_url,
                 probe_id, product_label, voting_confidence, probe_obj):
        """
        all params are saved on instance. sometimes with a manipulation.
        :param id: a product id. converted to: int.
        :param brand_code: not used.
        :param sample_type: SAMPLE TYPE: train,valid,test. converted to: string.
        :param brand_label: BRAND LABEL. converted to: int.
        :param form_factor_label: not used.
        :param mask: location of product in probe. converted to proper json.
        :param object_label: not used.
        :param patch_id: not used.
        :param patch_url: not used.
        :param probe_id: PROBE ID. converted to: string.
        :param product_label: PRODUCT LABEL. converted to: int.
        :param voting_confidence: not used.
        :param probe_obj: a pointer to the probe object this product belongs to.

        path: full path to cropped pic of product.
        relations: relations list.
        index_in_probe: serial index for product in probe.
        batch_id: all products in a probe are of a single batch. this is the PROBEID converted to int.
        """
        self.id = int(id)
        self.brand_code = str(brand_code)
        self.sample_type = str(sample_type)
        self.brand_label = brand_label_hasher.string_to_int(brand_label)
        self.form_factor_label = int(form_factor_label)
        mask = mask.replace('\'', '\"')  # fix json string
        self.mask = json.loads(mask)
        self.object_label = int(object_label)
        self.patch_id = int(patch_id)
        self.patch_url = str(patch_url)
        self.probe_id = str(probe_id)
        self.product_label = product_label_hasher.string_to_int(product_label)
        voting_confidence = voting_confidence.replace('\'', '\"')
        voting_confidence = voting_confidence.replace('u', '')
        self.voting_confidence = json.loads(voting_confidence)

        self.probe_obj = probe_obj
        self.path = products_dir + patch_url + ".jpg"
        self.relations = [[], []]
        self.index_in_probe = -1
        self.batch_id = probe_to_batch_hasher.string_to_int(self.probe_id)







