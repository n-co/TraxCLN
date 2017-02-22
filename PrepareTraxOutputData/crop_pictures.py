from config import *

def crop_probes():
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        for row in reader:
            probe_id = row[9]
            probe_path = probes_dir + probe_id + ".jpg"
            probe_img = ocv.imread(probe_path)

            mask = row[5].replace('\'', '\"')  # fix json string
            mask = json.loads(mask)
            cropped = probe_img[mask["y1"]:mask["y2"], mask["x1"]:mask["x2"]]
            cropped = ocv.resize(cropped, product_hw)  # resize image

            patch_url = row[8]


            # product = Product(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11])
            # cropped = product.features
            ocv.imwrite(products_dir + patch_url, cropped)

crop_probes()
