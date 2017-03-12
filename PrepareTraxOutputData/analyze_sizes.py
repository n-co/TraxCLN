from __future__ import division
from config import *



def analyze_sizes(csv_path):
    """
    according to a trax formatted csv file, makes jpg's of products from original pics.
    assumes csv is sorted by probe_id in order to be more effieicent.
    :param csv_path: path to a csv file.
    :return: nothing.
    """
    logging.info("analyze_sizes: Started.")
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        i = 0
        rs = []
        ws = []
        hs = []
        for row in reader:
            probe_id = row[9]
            probe_path = probes_dir + probe_id + ".jpg"


            mask = row[5].replace('\'', '\"')  # fix json string
            mask = json.loads(mask)
            w = mask["x2"]-mask["x1"]
            h = mask["y2"] - mask["y1"]
            r = h/w
            ws.append(w)
            hs.append(h)
            rs.append(r)
            if i % 2000 == 0:
                logging.debug("index now at: %d" %i)
            i += 1
        wavg = np.average(ws)
        wstd = np.std(ws)
        havg = np.average(hs)
        hstd = np.std(hs)
        ravg = np.average(rs)
        rstd = np.std(rs)
        logging.debug("width avg is: " + str(wavg))
        logging.debug("width std is: " + str(wstd))
        logging.debug("height avg is: " + str(havg))
        logging.debug("height std is: " + str(hstd))
        logging.debug("ratio avg is: " + str(ravg))
        logging.debug("ratio std is: " + str(rstd))
    logging.info("analyze_sizes: Ended.")


analyze_sizes(csv_path)
