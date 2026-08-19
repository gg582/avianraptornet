# BOLO

BOLO (*Birds Only Live Once*) is a motion-robust YOLO11n detector. It
combines selected AvianRaptorNet-inspired feature blocks with a lightweight
dual-stride motion-attention layer for blurred video frames.

## Components

- `modules.py` — `RaptorFoveal`, `FeatherBlock`, and
  `MotionStrideAttention`. The motion layer estimates coarse (stride 4) and
  fine (stride 2) motion energy, then gates a depthwise residual correction.
- `bolo11n.yaml` — YOLO11n-compatible model definition.
- `augment.py` — random linear motion blur augmentation.
- `prepare_data.py` — COCO annotations to YOLO labels conversion.
- `prepare_neatable.py` — 17-class desk-tidy dataset (TACO + office + COCO).
- `prepare_oid.py` — OpenImages download/conversion for classes missing from
  COCO (desk, plate, mug, pencil case, eraser, plus extra pen/ruler/scissors).
- `prepare_everyday.py` — 29-class everyday-objects dataset (neatable 17 +
  backpack, handbag, plate, bowl, spoon, fork, knife, mug, toothbrush, desk,
  pencil case, eraser). Writes `bolo_everyday.yaml`.
- `train_bolo.py` — BOLO training entry point.
- `export_bolo.py` — fused PyTorch state-dict, FP32 ONNX, and FP16 ONNX export.

## Quick start

Run all commands from the repository root with the project virtual
environment activated (or prefix commands with `venv/bin/`).

```bash
# Prepare labels and the small validation split.
python -m bolo.prepare_data --tiny

# Verify training and validation end to end.
python -m bolo.train_bolo --data bolo_tiny.yaml --epochs 2 --batch 16 --name bolo_smoke

# Train on the complete converted COCO subset.
python -m bolo.train_bolo --data bolo_coco20k.yaml --epochs 100 --batch 16 --name bolo_full

# Produce the final deployment files.
python -m bolo.export_bolo --weights runs/bolo/bolo_full/weights/best.pt
```

## Everyday-objects dataset (29 classes)

```bash
# Download OpenImages images for the non-COCO classes (one-time, ~2 GB).
python -m bolo.prepare_oid

# Build the merged dataset and bolo_everyday.yaml.
python -m bolo.prepare_everyday

# Train (~3 h on an RTX 3070 at 70 epochs).
python -m bolo.train_bolo --data bolo_everyday.yaml --epochs 70 --batch 16 --name bolo_everyday

# Export with a distinct prefix.
python -m bolo.export_bolo --weights runs/bolo/bolo_everyday/weights/best.pt --prefix bolo_everyday
```

The exporter writes these files to the repository root (prefix defaults to
`bolo11n_avian`, override with `--prefix`):

| Artifact | Purpose |
| --- | --- |
| `bolo11n_avian.pth` | Fused PyTorch state dictionary |
| `bolo11n_avian_fp32.onnx` | Simplified FP32 reference graph |
| `bolo11n_avian_fp16.onnx` | Fixed-shape FP16 ONNX graph for embedded inference |

## Design constraints

The custom layers use standard convolution, pooling, interpolation, and
elementwise ONNX operations only. BOLO does not modify the Ultralytics
installation: its custom modules are registered at runtime before the YAML
model is parsed.

FP16 ONNX uses a fixed input resolution (640 by default). Export it at the
resolution used by the target device, for example `--imgsz 320` for a smaller
deployment model.

## CIFAR-100 classifier

`BoloCIFAR100` reforms the proven `../aviannet` AvianRaptorNet-Fast CIFAR-100
model rather than training a new classifier from scratch. It retains the full
backbone and classifier, then inserts BOLO dual-stride motion correction after
the 256-channel stage and before classification. The added branches are
identity-initialized, so loading the original checkpoint preserves its logits
before fine-tuning.

```bash
# Downloads CIFAR-100 into data/ on first use.
python -m bolo.train_cifar100 --data ../aviannet/data --epochs 40 --batch-size 256 \
  --init-weights ../aviannet/avian_raptor_fast_best.pth

# Export the best classification checkpoint.
python -m bolo.export_cifar100 --weights bolo_cifar100_best.pth
```

The classification exporter produces `bolo_cifar100.pth`,
`bolo_cifar100_fp32.onnx`, and `bolo_cifar100_fp16.onnx`. Its input is
`[batch, 3, 32, 32]` and its output is 100 unnormalized class logits.
