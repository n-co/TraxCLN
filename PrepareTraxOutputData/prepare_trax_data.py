from config import *
from dbscan import dbscan


def compress_data(feats, labels, rel_list, train_ids, valid_ids, test_ids):
    f = open(pickle_path, 'wb')
    cPickle.dump((feats, labels, rel_list, train_ids, valid_ids, test_ids), f)
    f.close()

    in_file = file(pickle_path, 'rb')
    s = in_file.read()
    in_file.close()

    out_file = gzip.GzipFile(pickle_path+".gz", 'wb')
    out_file.write(s)
    out_file.close()

def load_data(path):
    f = gzip.open(path, 'rb')
    feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)

    return feats, labels, rel_list, train_ids, valid_ids, test_ids

def make_feats_and_labels(probes):
    ids = np.zeros(csv_length,dtype=int)
    labels = np.zeros(csv_length,dtype=int)
    feats = np.zeros((csv_length,product_size),dtype=type(np.ndarray))
    rel_list = np.zeros(csv_length,dtype=type(np.ndarray))
    for probe_id in probes:
        probe = probes[probe_id]
        for product in probe.products:
            # product.build_relations()
            product.relations = np.array(product.relations)
            # print "product #" + str(product.id) + ": " + str(product.relations[rel_left]) + str(product.relations[rel_right])
            labels[product.id] = product.brand_label  # product.product_label  # TODO: for now we took the brand_label which is an integer
            feats[product.id] = product.features
            rel_list[product.id] = product.relations
            ids[product.id] = product.id
    return ids, feats, labels, rel_list


def populate_probes(probes):
    for probe_id in probes:
        # print probe_id
        probe = probes[probe_id]
        shelves, noise = dbscan(probe.products, 1, eps, dist, sort_key)
        probe.set_shelves(shelves)
        # print map(lambda sh: map(lambda pr: pr.id, sh), shelves)
        # print map(lambda sh: map(lambda pr: pr.patch_url, sh), shelves)
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


def show_product_image(window_name, probes, probe_id, product_index):
    x = probes[probe_id].products[product_index].features
    ocv.imshow(window_name, x)
    ocv.waitKey(0)  # show plots


# def is_on_right(me, other):
#     if me == other:
#         return False
#     my_coords = me.mask
#     other_coodrds = other.mask
#     my_width = my_coords["x2"]-my_coords["x1"]
#     delta_x = np.abs(my_coords["x2"] - other_coodrds["x1"])
#     delta_y = np.abs(my_coords["y2"] - other_coodrds["y2"])
#     my_height = my_coords["y2"] - my_coords["y1"]
#     if delta_x <= gap_ratio_x * my_width and delta_y <= gap_ratio_y * my_height:
#         return True
#     else:
#         return False


def import_data():
    global csv_length  # declare that the global variable will be changed
    probes = {}
    sample_types = {
        "train": np.array([],dtype=int),
        "valid": np.array([],dtype=int),
        "test": np.array([],dtype=int)
    }
    probes_ids = glob.glob(probes_dir + "*.jpg")
    for i in range(0, len(probes_ids)):
        probes_ids[i] = probes_ids[i].strip(probes_dir)
        probes_ids[i] = probes_ids[i].strip(".jpg")
        probes[probes_ids[i]] = Probe(probes_ids[i])

    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)
        reader = csv.reader(lines)
        csv_length = 0
        for row in reader:
            csv_length += 1
            product = Product(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], probes[row[9]])
            product.index_in_probe = len(probes[product.probe_id].products)
            probes[product.probe_id].products = np.append(probes[product.probe_id].products, product)
            sample_types[product.sample_type] = np.append(sample_types[product.sample_type], product.id)
    return {
        "probes": probes,
        "train": sample_types["train"],
        "valid": sample_types["valid"],
        "test": sample_types["test"]
    }

raw_data = import_data()
train_ids = raw_data["train"]
valid_ids = raw_data["valid"]
test_ids = raw_data["test"]
probes = raw_data["probes"]
populate_probes(probes)
ids, feats, labels, rel_list = make_feats_and_labels(probes)

compress_data(feats, labels, rel_list, train_ids, valid_ids, test_ids)

feats, labels, rel_list, train_ids, valid_ids, test_ids = load_data(pickle_path + ".gz")