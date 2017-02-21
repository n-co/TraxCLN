from config import *

def crop_probes():
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        for row in reader:
            product = Product(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11])
            cropped = product.features
            ocv.imwrite(products_dir + product.patch_url, cropped)

crop_probes()
