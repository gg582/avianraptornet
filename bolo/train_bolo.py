"""BOLO training pipeline.

Trains the BOLO (AvianRaptorNet x YOLO11n) detector with synthetic
motion-blur augmentation. Defaults are tuned for an RTX 3070 8 GB.

Usage:
    # Smoke test on the tiny split (validates the whole pipeline)
    python -m bolo.train_bolo --data bolo_tiny.yaml --epochs 2 --name bolo_smoke

    # Full training on the converted COCO 20k subset
    python -m bolo.train_bolo --data bolo_coco20k.yaml --epochs 100 --batch 16 --name bolo_full
"""

import argparse
from pathlib import Path

import bolo  # noqa: F401  (registers custom modules)
from bolo.augment import BoloDataset

ROOT = Path(__file__).resolve().parent.parent


def build_trainer(overrides, motion_blur=0.3):
    from copy import copy

    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import colorstr
    from ultralytics.utils.torch_utils import unwrap_model

    class BoloTrainer(DetectionTrainer):
        """DetectionTrainer that builds BoloDataset (motion-blur aware)."""

        def build_dataset(self, img_path, mode="train", batch=None):
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            # Custom hyp key: not a valid YOLO arg, so keep it on a copy of the
            # namespace instead of self.args (the validator re-validates self.args).
            hyp = copy(self.args)
            hyp.motion_blur = self.motion_blur_p
            cfg = self.args
            return BoloDataset(
                img_path=img_path,
                imgsz=cfg.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=hyp,
                rect=cfg.rect or mode == "val",
                cache=cfg.cache or None,
                single_cls=cfg.single_cls or False,
                stride=gs,
                pad=0.0 if mode == "train" else 0.5,
                prefix=colorstr(f"{mode}: "),
                task=cfg.task,
                classes=cfg.classes,
                data=self.data,
                fraction=cfg.fraction if mode == "train" else 1.0,
            )

    trainer = BoloTrainer(overrides=overrides)
    trainer.motion_blur_p = motion_blur
    return trainer


def main():
    ap = argparse.ArgumentParser(description="Train BOLO detector")
    ap.add_argument("--data", default="bolo_tiny.yaml", help="dataset yaml (in bolo/ or ultralytics-known)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--motion-blur", type=float, default=0.3, help="motion blur probability (0 disables)")
    ap.add_argument("--pretrained", default="", help="optional .pt for partial weight transfer (e.g. yolo11n.pt)")
    ap.add_argument("--name", default="bolo_train")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    data = args.data
    if not (ROOT / data).exists() and (Path(__file__).parent / data).exists():
        data = str(Path(__file__).parent / data)

    overrides = {
        "model": str(Path(__file__).parent / "bolo11n.yaml"),
        "data": data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "project": str(ROOT / "runs" / "bolo"),
        "name": args.name,
        "optimizer": "auto",
        "cos_lr": True,
        "amp": True,
        "pretrained": args.pretrained if args.pretrained else False,
    }

    trainer = build_trainer(overrides, motion_blur=args.motion_blur)
    trainer.train()
    print(f"best weights: {trainer.best}")


if __name__ == "__main__":
    main()
