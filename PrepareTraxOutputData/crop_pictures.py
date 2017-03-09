from config import *


def crop_probes(csv_path, probes_dir, products_dir, product_hw):
    """
    according to a trax formatted csv file, makes jpg's of products from original pics.
    assumes csv is sorted by probe_id in order to be more effieicent.
    :param csv_path: path to a csv file.
    :param probes_dir: directory in which source probes are located.
    :param products_dir: directory to place cropped products.
    :param product_hw: height and width of product, after resize.
    :return: nothing.
    """
    logging.info("crop_probes: Started.")
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        i = 1
        prev_probe_id = None
        probe_img = None;
        for row in reader:
            probe_id = row[9]
            probe_path = probes_dir + probe_id + ".jpg"

            if probe_id != prev_probe_id:
                probe_img = ocv.imread(probe_path)
                prev_probe_id = probe_id

            mask = row[5].replace('\'', '\"')  # fix json string
            mask = json.loads(mask)
            cropped = probe_img[mask["y1"]:mask["y2"], mask["x1"]:mask["x2"]]
            cropped = ocv.resize(cropped, product_hw)  # resize image

            patch_url = row[8]
            # product = Product(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11])
            # cropped = product.features
            ocv.imwrite(products_dir + patch_url, cropped)
            if i % 1000 == 0:
                logging.debug("cropped 1000 products. i is: %d " % i)
            i += 1
    logging.info("crop_probes: Ended.")

crop_probes(csv_path, probes_dir, products_dir, product_hw)
