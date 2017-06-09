from __future__ import division
from config import *
import numpy as np
import itertools
import csv
import json


def analyze_sizes(csv_path):
    """
    goes over a csv file, to recover statistics over it.
    :param csv_path: path to a csv file.
    :return: nothing.
    """
    logging.debug("analyze_sizes: Started.")
    f = open(csv_path, 'r')

    lines = itertools.islice(f, 1, None)  # ignore header line
    reader = csv.reader(lines)
    i = 0
    rs = []
    ws = []
    hs = []
    for row in reader:
        mask = row[5].replace('\'', '\"')  # fix json string
        mask = json.loads(mask)
        w = mask["x2"] - mask["x1"]
        h = mask["y2"] - mask["y1"]
        r = h / w
        ws.append(w)
        hs.append(h)
        rs.append(r)
        if i % 2000 == 0:
            logging.debug("index now at: %d" % i)
        i += 1
    wavg = np.average(ws)
    wstd = np.std(ws)
    havg = np.average(hs)
    hstd = np.std(hs)
    ravg = np.average(rs)
    rstd = np.std(rs)
    logging.info("width avg is: " + str(wavg))
    logging.info("width std is: " + str(wstd))
    logging.info("height avg is: " + str(havg))
    logging.info("height std is: " + str(hstd))
    logging.info("ratio avg is: " + str(ravg))
    logging.info("ratio std is: " + str(rstd))

    f.close()

    logging.debug("analyze_sizes: Ended.")


analyze_sizes(csv_path)
