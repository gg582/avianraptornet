"""BOLO model build and ultralytics registration.

All custom modules are channel-preserving, so the `parse_model` else-branch
(`c2 = ch[f]`) keeps the channel bookkeeping intact. Instead of modifying
site-packages, the classes are injected into the `ultralytics.nn.tasks`
module globals so YAML name resolution works.
"""

from pathlib import Path

import ultralytics.nn.tasks as tasks

from bolo.modules import BioMish, ChannelAttention, FeatherBlock, MotionStrideAttention, RaptorFoveal

BOLO_YAML = Path(__file__).resolve().parent / "bolo11n.yaml"

_BOLO_MODULES = (BioMish, ChannelAttention, FeatherBlock, MotionStrideAttention, RaptorFoveal)


def register_bolo_modules():
    """Inject custom modules into the ultralytics tasks globals for YAML parsing."""
    for m in _BOLO_MODULES:
        setattr(tasks, m.__name__, m)


def build_bolo(cfg=None, scale="n", verbose=True):
    """Build the BOLO DetectionModel."""
    register_bolo_modules()
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    d = yaml_model_load(str(cfg or BOLO_YAML))
    d["scale"] = scale
    return DetectionModel(d, ch=3, verbose=verbose)
