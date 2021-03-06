from config import *
from PIL import Image, ImageDraw
import itertools
import csv
import json


def frame_products():
    prev_probe_id = None
    probe_img = None
    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        for row in reader:
            probe_id = row[9]

            if probe_id != prev_probe_id:
                save_image(probe_img, prev_probe_id)
                prev_probe_id = probe_id
                probe_img = Image.open(probes_dir + probe_id + ".jpg")
                logging.debug("probe id: %s" % probe_id)

            mask = row[5].replace('\'', '\"')  # fix json string
            mask = json.loads(mask)
            x1 = mask["x1"]
            x2 = mask["x2"]
            y1 = mask["y1"]
            y2 = mask["y2"]
            draw = ImageDraw.Draw(probe_img)
            draw.line((x1, y1, x2, y1), fill=255, width=10)  # top line
            draw.line((x2, y1, x2, y2), fill=255, width=10)  # right line
            draw.line((x1, y1, x1, y2), fill=255, width=10)  # left line
            draw.line((x1, y2, x2, y2), fill=255, width=10)  # bottom line
    save_image(probe_img, prev_probe_id)  # save last image


def save_image(img, probe_id):
    if img is not None:
        img.save(framed_probes_dir + probe_id + ".jpg")


frame_products()
