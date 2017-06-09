from config import *
import dbscan
import classes
import numpy as np
import cPickle
import gzip
import glob
import csv
import itertools




def compress_to_gzip_file(labels, rel_list, train_ids, valid_ids, test_ids, paths, batchs):
    logging.debug('compress_to_gzip_file - Started.')
    f = open(pickle_path, 'wb')
    cPickle.dump((labels, rel_list, train_ids, valid_ids, test_ids, paths, batchs), f)
    f.close()

    in_file = file(pickle_path, 'rb')
    s = in_file.read()
    in_file.close()

    out_file = gzip.GzipFile(pickle_path+".gz", 'wb')
    out_file.write(s)
    out_file.close()
    logging.debug('compress_to_gzip_file - Ended.')


def load_gzip_file(path):
    logging.debug('load_gzip_file - Started')
    f = gzip.open(path, 'rb')
    labels, rel_list, train_ids, valid_ids, test_ids, paths, batchs = cPickle.load(f)
    logging.debug('load_gzip_file - Ended.')
    return labels, rel_list, train_ids, valid_ids, test_ids, paths, batchs


def format_ids_feats_labels_rel_list(probes):
    logging.debug('format_ids_feats_labels_rel_list - Started.')
    logging.info('format_ids_feats_labels_rel_list: csv_length is: %d.' % csv_length)
    ids = np.zeros(csv_length, dtype=int)
    labels = np.zeros(csv_length, dtype=int)
    # feats = np.zeros((csv_length, product_height, product_width, product_channels), dtype=type(np.ndarray))
    paths = np.zeros(csv_length, dtype=object)
    rel_list = np.zeros(csv_length, dtype=type(np.ndarray))
    batchs = np.zeros(csv_length, dtype=object)
    for probe_id in probes:
        probe = probes[probe_id]
        for product in probe.products:
            product.relations = np.array(product.relations)
            labels[product.id] = product.product_label
            paths[product.id] = str(products_dir + product.patch_url)
            rel_list[product.id] = product.relations
            ids[product.id] = product.id
            batchs[product.id] = product.batch_id
    logging.debug('format_ids_feats_labels_rel_list - Ended.')
    return ids, labels, rel_list, paths, batchs


def populate_probes(probes):
    logging.debug('Populate Probes - Started.')
    for probe_id in probes:
        probe = probes[probe_id]
        shelves, noise = dbscan.dbscan(probe.products, 1, eps, dist, sort_key)
        probe.set_shelves(shelves)
        # logging.debug(str(map(lambda sh: map(lambda pr: pr.id, sh), shelves)))
        # logging.debug(str(map(lambda sh: map(lambda pr: pr.patch_url, sh), shelves)))
        probe.build_relations()

        # build matrices, not sure if it is necessary
        curr = probe.products
        n = len(curr)
        rights = np.zeros((n, n))
        lefts = np.zeros((n, n))
        for i in range(0, n):
            product = curr[i]
            for j in range(0, n):
                neighbour = curr[j].id
                rights[i][j] = int(neighbour in product.relations[rel_right])
                lefts[i][j] = int(neighbour in product.relations[rel_left])
        probe.set_rights(rights)
        probe.set_lefts(lefts)
    logging.debug('Populate Probes - Ended.')






def import_data():
    logging.debug("import_data - Started.")
    global csv_length  # declare that the variable is global so it can be modified.
    probes = {}
    sample_types = {
        "train": np.array([], dtype=int),
        "valid": np.array([], dtype=int),
        "test": np.array([], dtype=int)
    }
    probes_ids = glob.glob(probes_dir + "*.jpg")
    amount_of_probes = len(probes_ids)
    logging.info("going over %d probes." % amount_of_probes)
    for i in range(0, amount_of_probes):
        probes_ids[i] = probes_ids[i].strip(probes_dir)
        probes_ids[i] = probes_ids[i].strip(".jpg")
        probes[probes_ids[i]] = classes.Probe(probes_ids[i])
    logging.info("going over products in csv file.")
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)
        reader = csv.reader(lines)
        csv_length = 0
        i = 0
        for row in reader:
            csv_length += 1
            probe_id = row[9]
            sample_type = get_sample_type(probe_id)
            # TODO: remove sample type col fromsource csv file
            product = classes.Product(id=row[0],
                              brand_code=row[1],
                              sample_type=sample_type,
                              brand_label=row[3],
                              form_factor_label=row[4],
                              mask=row[5],
                              object_label=row[6],
                              patch_id=row[7],
                              patch_url=row[8],
                              probe_id=row[9],
                              product_label=row[10],
                              voting_confidence=row[11],
                              probe_obj=probes[row[9]])
            product.index_in_probe = len(probes[product.probe_id].products)
            probes[product.probe_id].products = np.append(probes[product.probe_id].products, product)
            sample_types[product.sample_type] = np.append(sample_types[product.sample_type], product.id)

            if should_i_stop(probe_id):
                break
            i += 1
    logging.debug("import_data - Ended.")
    return probes, sample_types["train"], sample_types["valid"], sample_types['test']


def prepare_trax_data():
    logging.debug("prepare_trax_data - Started.")
    probes, train_ids, valid_ids, test_ids = import_data()
    populate_probes(probes)
    ids, labels, rel_list, paths, batchs = format_ids_feats_labels_rel_list(probes)
    compress_to_gzip_file(labels, rel_list, train_ids, valid_ids, test_ids, paths, batchs)
    logging.info("length of train: %d valid: %d test: %d " % (len(train_ids), len(valid_ids), len(test_ids)))
    logging.info("number of different labels: %d" % np.max(labels))
    logging.debug("train labels:")
    logging.debug(str(np.bincount(labels[train_ids])))
    logging.debug("test labels:")
    logging.debug(str(np.bincount(labels[test_ids])))
    logging.debug("prepare_trax_data- Ended.")

prepare_trax_data()
