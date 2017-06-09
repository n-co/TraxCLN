from config import *
import numpy as np
import cv2 as ocv
import json
import itertools
import csv

paint_pixel = None  # will be a function that returns single pixel [r, g, b]


def crop_probes(csv_path, probes_dir, products_dir, product_wh):
    """
    according to a trax formatted csv file, makes jpg's of products from original pics.
    assumes csv is sorted by probe_id in order to be more effieicent.
    :param csv_path: path to a csv file.
    :param probes_dir: directory in which source probes are located.
    :param products_dir: directory to place cropped products.
    :param product_wh: height and width of product, after resize.
    :return: nothing.
    """
    global paint_pixel
    logging.info("crop_probes: Started.")

    if pad_type == 'noise':
        paint_pixel = make_random_pixel

    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        i = 1
        prev_probe_id = None
        probe_img = None
        for row in reader:
            probe_id = row[9]
            probe_path = probes_dir + probe_id + ".jpg"

            if probe_id != prev_probe_id:
                probe_img = ocv.imread(probe_path)
                prev_probe_id = probe_id

            mask = row[5].replace('\'', '\"')  # fix json string
            mask = json.loads(mask)
            cropped = probe_img[mask["y1"]:mask["y2"], mask["x1"]:mask["x2"]]

            cropped = scaled_resize(cropped)
            cropped = pad_product(cropped)

            patch_url = row[8]
            # product = Product(row[0], row[1], row[2], row[3], row[4], row[5],
            #                   row[6], row[7], row[8], row[9], row[10], row[11])
            # cropped = product.features
            ocv.imwrite(products_dir + patch_url, cropped)
            if i % 1000 == 0:
                logging.debug("cropped 1000 products. i is: %d " % i)
            i += 1
    logging.info("crop_probes: Ended.")


def scaled_resize(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_ratio = float(img_height) / float(img_width)
    product_ratio = float(product_height) / float(product_width)

    if img_width <= product_width and img_height <= product_height:
        return img  # no need to resize

    if img_ratio > product_ratio:
        # img is "slim and tall"
        new_width = img_width * (float(product_height) / float(img_height))
        new_width = int(new_width)
        img = ocv.resize(img, (new_width, product_height))
    else:
        # img is "fat and short"
        new_height = img_height * (float(product_width) / float(img_width))
        new_height = int(new_height)
        img = ocv.resize(img, (product_width, new_height))
    return img


def pad_product(img):
    img_height = img.shape[0]
    img_width = img.shape[1]

    top_pad = (product_height - img_height) / 2
    bottom_pad = product_height - img_height - top_pad
    right_pad = (product_width - img_width) / 2
    left_pad = product_width - img_width - right_pad
    # pad with black as default
    img = ocv.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, ocv.BORDER_CONSTANT, value=[0, 0, 0])

    if paint_pixel is not None:
        for h in range(len(img)):
            for w in range(len(img[h])):
                if not(left_pad <= w < left_pad + img_width and top_pad <= h < top_pad + img_height):
                    img[h, w] = paint_pixel()
    return img


def make_random_pixel():
    return (np.random.rand(3) * 256).astype(int)

print "csv path:     " + csv_path
print "probes dir:   " + probes_dir
print "products dir: " + products_dir
print "product W/H:  " + str(product_wh)
crop_probes(csv_path, probes_dir, products_dir, product_wh)
print "csv path:     " + csv_path
print "probes dir:   " + probes_dir
print "products dir: " + products_dir
print "product W/H:  " + str(product_wh)
