from config import *
from PIL import Image, ImageDraw


def frame_products():
    probes_ids = glob.glob(probes_dir + "*.jpg")
    probes = {}
    for i in range(0, len(probes_ids)):
        probes_ids[i] = probes_ids[i].strip(probes_dir)
        probes_ids[i] = probes_ids[i].strip(".jpg")
        probes[probes_ids[i]] = Image.open(probes_dir + probes_ids[i] + ".jpg")

    with open(csv_path, 'r') as f:
        lines = itertools.islice(f, 1, None)  # ignore header line
        reader = csv.reader(lines)
        for row in reader:
            probe_id = row[9]

            mask = row[5].replace('\'', '\"')  # fix json string
            mask = json.loads(mask)
            x1 = mask["x1"]
            x2 = mask["x2"]
            y1 = mask["y1"]
            y2 = mask["y2"]
            draw = ImageDraw.Draw(probes[probe_id])
            draw.line((x1, y1, x2, y1), fill=255, width=10)  # top line
            draw.line((x2, y1, x2, y2), fill=255, width=10)  # right line
            draw.line((x1, y1, x1, y2), fill=255, width=10)  # left line
            draw.line((x1, y2, x2, y2), fill=255, width=10)  # bottom line

    for id in probes:
        probes[id].save(framed_probes_dir + id + ".jpg")



frame_products()
