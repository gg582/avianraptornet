"""Synthetic motion-blur augmentation for BOLO training.

Applies a random linear motion-blur kernel to the composed training image
(after mosaic / affine, before Format), so the MotionStrideAttention layer
learns to detect and correct blurred regions. Boxes are unaffected because
the blur is a pure image-space degradation.
"""

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset


class MotionBlur:
    """Random linear motion blur transform for the ultralytics pipeline.

    Args:
        p (float): Probability of applying the blur to an image.
        kmin (int): Minimum kernel length (odd, >= 3).
        kmax (int): Maximum kernel length (odd).
    """

    def __init__(self, p=0.3, kmin=5, kmax=31):
        self.p = p
        self.kmin = max(3, kmin | 1)  # force odd
        self.kmax = max(self.kmin, kmax | 1)

    def _kernel(self, rng):
        k = rng.integers(self.kmin // 2, self.kmax // 2 + 1) * 2 + 1
        angle = float(rng.uniform(0.0, 180.0))
        kern = np.zeros((k, k), dtype=np.float32)
        kern[k // 2, :] = 1.0
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kern = cv2.warpAffine(kern, rot, (k, k))
        s = kern.sum()
        return kern / s if s > 0 else None

    def __call__(self, labels):
        rng = np.random.default_rng()
        if rng.random() >= self.p:
            return labels
        img = labels.get("img")
        if img is None:
            return labels
        kern = self._kernel(rng)
        if kern is not None:
            labels["img"] = cv2.filter2D(img, -1, kern)
        return labels


class BoloDataset(YOLODataset):
    """YOLODataset with synthetic motion blur injected before Format."""

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        p = float(getattr(hyp, "motion_blur", 0.0) or 0.0)
        if self.augment and p > 0.0:
            transforms.insert(-1, MotionBlur(p=p))
        return transforms
