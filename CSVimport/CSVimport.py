from config import *


def populate_probes(probes):
    for probe_id in probes:
        probe = probes[probe_id]
        curr = probe.products
        n = len(curr)
        rights = np.zeros((n, n))
        for i in range(0, n):
            meee = curr[i]
            for j in range(0, n):
                otherrrr = curr[j]
                rights[i][j] = is_on_right(meee, otherrrr)
        probe.rights = rights
        probe.lefts = rights.transpose()


def show_product_image(window_name,product_list, probe_id,product_index):
    x = product_list[probe_id][product_index].features
    ocv.imshow(window_name,x)
    ocv.waitKey(0)  # show plots


def is_on_right(me, other):
    if me == other:
        return False
    my_coords = me.mask
    other_coodrds = other.mask
    my_width = my_coords["x2"]-my_coords["x1"]
    delta_x = np.abs(my_coords["x2"] - other_coodrds["x1"])
    delta_y = np.abs(my_coords["y2"] - other_coodrds["y2"])
    my_height = my_coords["y2"] - my_coords["y1"]
    if delta_x <= gap_ratio_x * my_width and delta_y <= gap_ratio_y * my_height:
        return True
    else:
        return False


def import_data():
    probes = {}
    sample_types = {
        "train": [],
        "valid": [],
        "test": []
    }
    probes_ids = glob.glob(probes_dir + "*.jpg")
    for i in range(0, len(probes_ids)):
        feats = ocv.imread(probes_ids[i])
        probes_ids[i] = probes_ids[i].strip(probes_dir)
        probes_ids[i] = probes_ids[i].strip(".jpg")
        probes[probes_ids[i]] = Probe(probes_ids[i], feats)
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)
        reader = csv.reader(lines)
        for row in reader:
            product = Product(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11],probes[row[9]])
            probes[product.probe_id].products.append(product)
            sample_types[product.sample_type].append(product.id)
    return {
        "probes": probes,
        "train": sample_types["train"],
        "valid": sample_types["valid"],
        "test": sample_types["test"]
    }

raw_data = import_data()
probes = raw_data["probes"]
print raw_data
print probes['10521992'].products[0].features.shape
print probes['10521992'].products[0].product_label
# products = raw_data["products"]
# probes = raw_data["probes"]
# # show_product_image("noam", products, '10521992', 2)
#
# print 2

all_labels = []
all_features = []
for probe_id in probes:
    probe = probes[probe_id]
    for product in probe.products:
        all_labels.append(product.product_label)
        all_features.append(product.features)
print all_labels
print all_features
populate_probes(probes)
