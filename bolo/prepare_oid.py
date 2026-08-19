"""Download OpenImages subsets for classes missing from COCO/TACO/office.

Streams kept to a minimum: expects ``oid_train_filtered.csv`` (already
pre-filtered from the full oidv6 train annotations, see README) and the
local ``validation-annotations-bbox.csv`` for the val split.

Outputs a YOLO dataset under ``data/openimages_office/yolo`` with classes:
    desk, plate, mug, pencil case, eraser, pen, ruler, scissors

Usage:
    python -m bolo.prepare_oid
"""

import csv
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OID = ROOT / "data" / "openimages_office"
OUT = OID / "yolo"

S3 = "https://s3.amazonaws.com/open-images-dataset"

# label id -> (class name, max boxes for train, max boxes for val)
CLASSES = {
    "/m/01y9k5": ("desk", 2000, 250),
    "/m/050gv4": ("plate", 1500, 250),
    "/m/02jvh9": ("mug", 1500, 250),
    "/m/05676x": ("pencil case", 1000, 200),
    "/m/02fh7f": ("eraser", 1000, 200),
    "/m/0k1tl": ("pen", 1200, 200),
    "/m/0hdln": ("ruler", 1000, 200),
    "/m/01lsmm": ("scissors", 1000, 200),
}
NAME2IDX = {name: i for i, (name, *_rest) in enumerate(CLASSES.values())}

TRAIN_CSV = OID / "oid_train_filtered.csv"
VAL_CSV = OID / "validation-annotations-bbox.csv"


def load_boxes(csv_path, label_ids):
    """Return {image_id: [(label_id, xmin, xmax, ymin, ymax), ...]}."""
    boxes = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lid = row["LabelName"]
            if lid not in label_ids:
                continue
            if row.get("IsGroupOf") == "1" or row.get("Confidence") != "1":
                continue
            boxes.setdefault(row["ImageID"], []).append(
                (lid, float(row["XMin"]), float(row["XMax"]),
                 float(row["YMin"]), float(row["YMax"]))
            )
    return boxes


def select_images(boxes, caps, rng):
    """Pick images until per-class box caps are met.

    Returns {image_id: [boxes]} with boxes restricted to target classes.
    """
    image_ids = list(boxes)
    rng.shuffle(image_ids)
    counts = {lid: 0 for lid in caps}
    picked = {}
    for iid in image_ids:
        keep = []
        for b in boxes[iid]:
            lid = b[0]
            if counts[lid] < caps[lid]:
                keep.append(b)
                counts[lid] += 1
        if keep:
            picked[iid] = keep
        if all(counts[lid] >= caps[lid] for lid in caps):
            break
    return picked, counts


def download(iid, split):
    dst = OID / f"{split}_images" / f"{iid}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return iid, True
    s3_split = {"val": "validation"}.get(split, split)
    url = f"{S3}/{s3_split}/{iid}.jpg"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r, \
                    open(dst, "wb") as f:
                f.write(r.read())
            return iid, True
        except Exception:
            time.sleep(1 + attempt)
    return iid, False


def build_split(csv_path, split, caps, rng):
    boxes = load_boxes(csv_path, set(caps))
    picked, counts = select_images(boxes, caps, rng)
    print(f"[{split}] candidate images={len(boxes)} picked={len(picked)}")
    for lid, (name, *_rest) in CLASSES.items():
        print(f"    {name:12s} boxes={counts[lid]}")

    img_out = OUT / split / "images"
    lab_out = OUT / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    (OID / f"{split}_images").mkdir(exist_ok=True)

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda iid: download(iid, split), picked))
    ok = {iid for iid, good in results if good}
    print(f"[{split}] downloaded {len(ok)}/{len(picked)}")

    from PIL import Image
    n_img = 0
    for iid in sorted(ok):
        src = OID / f"{split}_images" / f"{iid}.jpg"
        try:
            w, h = Image.open(src).size
        except Exception:
            continue
        lines = []
        for lid, xmin, xmax, ymin, ymax in picked[iid]:
            name = CLASSES[lid][0]
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{NAME2IDX[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue
        link = img_out / f"oid_{iid}.jpg"
        if not link.exists():
            link.symlink_to(src.resolve())
        (lab_out / f"oid_{iid}.txt").write_text("\n".join(lines) + "\n")
        n_img += 1
    print(f"[{split}] wrote {n_img} images to {OUT / split}")


def main():
    rng = random.Random(0)
    train_caps = {lid: v[1] for lid, v in CLASSES.items()}
    val_caps = {lid: v[2] for lid, v in CLASSES.items()}
    if TRAIN_CSV.exists():
        build_split(TRAIN_CSV, "train", train_caps, rng)
    else:
        print(f"skip train: {TRAIN_CSV} not found")
    build_split(VAL_CSV, "val", val_caps, rng)


if __name__ == "__main__":
    main()
