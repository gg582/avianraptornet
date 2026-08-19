"""Build the extended everyday-objects detection dataset.

Extends the 17-class neatable desk-tidy set (see bolo/prepare_neatable.py)
with 12 everyday-object classes, for 29 classes total:

    existing 17 + backpack, handbag, plate, bowl, spoon, fork, knife,
    mug, toothbrush, desk, pencil case, eraser

Sources:
- TACO (roboflow export): bottle/can/cup/paper/trash
- Office supplies (kaggle): pen, highlighter, glue, scissors, tape, ruler
- COCO 20k subset: desk objects + backpack/handbag/fork/knife/spoon/bowl/toothbrush
- OpenImages (bolo/prepare_oid.py): desk, plate, mug, pencil case, eraser
  plus extra pen/ruler/scissors boxes (these were scarce before)

Images are symlinked; labels are rewritten with remapped class ids.
Also writes ``bolo/bolo_everyday.yaml`` for training.

Usage:
    python -m bolo.prepare_oid       # if OID yolo data not built yet
    python -m bolo.prepare_everyday
"""

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "everyday_objects"
YAML_OUT = Path(__file__).resolve().parent / "bolo_everyday.yaml"

NAMES = [
    "bottle", "can", "cup", "paper", "trash",
    "pen", "highlighter", "glue", "scissors", "tape", "ruler",
    "book", "keyboard", "mouse", "cell phone", "clock", "laptop",
    "backpack", "handbag", "plate", "bowl", "spoon", "fork", "knife",
    "mug", "toothbrush", "desk", "pencil case", "eraser",
]
IDX = {name: i for i, name in enumerate(NAMES)}

# --- source mappings -------------------------------------------------------
TACO = ROOT / "data" / "kaggle_raw" / "taco"
TACO_MAP = {
    2: IDX["bottle"],   # Bottle
    4: IDX["can"],      # Can
    6: IDX["cup"],      # Cup
    11: IDX["paper"],   # Paper
}
# every other TACO class -> trash
TACO_TRASH = set(range(18)) - set(TACO_MAP)

OFFICE = ROOT / "data" / "kaggle_raw" / "office" / "data"
OFFICE_MAP = {
    15: IDX["tape"],        # tape dispenser
    16: IDX["highlighter"],
    17: IDX["glue"],        # glue stick
    18: IDX["pen"],
    19: IDX["scissors"],
    20: IDX["ruler"],       # set square
    21: IDX["tape"],        # correction tape
}

COCO = ROOT / "data" / "coco_subset_20k"
COCO_MAP = {
    39: IDX["bottle"],
    41: IDX["cup"],
    76: IDX["scissors"],
    73: IDX["book"],
    66: IDX["keyboard"],
    64: IDX["mouse"],
    67: IDX["cell phone"],
    74: IDX["clock"],
    63: IDX["laptop"],
    24: IDX["backpack"],
    26: IDX["handbag"],
    42: IDX["fork"],
    43: IDX["knife"],
    44: IDX["spoon"],
    45: IDX["bowl"],
    79: IDX["toothbrush"],
}

OID = ROOT / "data" / "openimages_office" / "yolo"
# oid yolo class order (bolo/prepare_oid.py): desk, plate, mug, pencil case,
# eraser, pen, ruler, scissors
OID_MAP = {
    0: IDX["desk"],
    1: IDX["plate"],
    2: IDX["mug"],
    3: IDX["pencil case"],
    4: IDX["eraser"],
    5: IDX["pen"],
    6: IDX["ruler"],
    7: IDX["scissors"],
}

COCO_MAX_TRAIN = 6000   # cap so COCO does not swamp the specialist sets
COCO_VAL = 500
TACO_VAL_MAX = 400


def remap_file(label_path, mapping, default=None):
    """Return remapped label text; None if no boxes survive."""
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls in mapping:
            out.append(f"{mapping[cls]} " + " ".join(parts[1:]))
        elif default is not None and cls in default:
            out.append(f"{IDX['trash']} " + " ".join(parts[1:]))
    return "\n".join(out) + "\n" if out else None


def safe_name(prefix, path):
    return f"{prefix}_{path.name.replace(' ', '_')}"


def add_split(split_entries, split):
    img_dir = OUT / split / "images"
    lab_dir = OUT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_img, src_lab, name, text in split_entries:
        img_link = img_dir / name
        if not img_link.exists():
            img_link.symlink_to(src_img)
        (lab_dir / (Path(name).stem + ".txt")).write_text(text)
        count += 1
    return count


def collect(src_root, img_split, prefix, mapping, default=None, limit=None, rng=None):
    entries = []
    img_dir = src_root / img_split / "images"
    lab_dir = src_root / img_split / "labels"
    files = sorted(img_dir.iterdir())
    if rng:
        rng.shuffle(files)
    for img in files:
        lab = lab_dir / (img.stem + ".txt")
        if not lab.exists():
            continue
        text = remap_file(lab, mapping, default)
        if text is None:
            continue
        entries.append((img, lab, safe_name(prefix, img), text))
        if limit and len(entries) >= limit:
            break
    return entries


def collect_flat_coco(rng):
    """coco_subset_20k stores images/ and labels/ flat at the root."""
    entries = []
    img_dir, lab_dir = COCO / "images", COCO / "labels"
    files = sorted(img_dir.iterdir())
    rng.shuffle(files)
    for img in files:
        lab = lab_dir / (img.stem + ".txt")
        if not lab.exists():
            continue
        text = remap_file(lab, COCO_MAP)
        if text is None:
            continue
        entries.append((img, lab, safe_name("coco", img), text))
    return entries


def box_stats(entries):
    from collections import Counter
    counts = Counter()
    for _img, _lab, _name, text in entries:
        for line in text.splitlines():
            counts[NAMES[int(line.split()[0])]] += 1
    return counts


def main():
    rng = random.Random(0)
    if OUT.exists():
        shutil.rmtree(OUT)

    # --- train ---
    taco_train = collect(TACO, "train", "taco", TACO_MAP, default=TACO_TRASH)
    office_train = collect(OFFICE, "train", "office", OFFICE_MAP)
    oid_train = collect(OID, "train", "oid", OID_MAP) if OID.is_dir() else []
    coco_all = collect(COCO, ".", "coco", COCO_MAP, rng=rng) \
        if (COCO / "images").is_dir() else []
    # coco_subset_20k has flat images/ + labels/ (no split dirs)
    if not coco_all:
        coco_all = collect_flat_coco(rng)
    coco_val = coco_all[:COCO_VAL]
    coco_train = coco_all[COCO_VAL:COCO_VAL + COCO_MAX_TRAIN]

    # --- val ---
    taco_val = collect(TACO, "valid", "taco", TACO_MAP, default=TACO_TRASH,
                       limit=TACO_VAL_MAX, rng=rng)
    office_val = collect(OFFICE, "validation", "office", OFFICE_MAP)
    oid_val = collect(OID, "val", "oid", OID_MAP) if OID.is_dir() else []

    train_entries = taco_train + office_train + oid_train + coco_train
    val_entries = taco_val + office_val + oid_val + coco_val
    n_train = add_split(train_entries, "train")
    n_val = add_split(val_entries, "val")

    yaml_lines = [
        "# BOLO everyday-objects dataset (generated by bolo/prepare_everyday.py)",
        f"path: {OUT}",
        "train: train/images",
        "val: val/images",
        f"nc: {len(NAMES)}",
        "names:",
    ]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(NAMES)]
    YAML_OUT.write_text("\n".join(yaml_lines) + "\n")

    print(f"train: {n_train} images  (taco={len(taco_train)}, "
          f"office={len(office_train)}, oid={len(oid_train)}, "
          f"coco={len(coco_train)})")
    print(f"val:   {n_val} images  (taco={len(taco_val)}, "
          f"office={len(office_val)}, oid={len(oid_val)}, coco={len(coco_val)})")
    stats = box_stats(train_entries)
    print("train box distribution:")
    for name in NAMES:
        print(f"  {name:12s} {stats[name]}")
    print(f"wrote {YAML_OUT}")


if __name__ == "__main__":
    main()
