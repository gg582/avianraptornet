"""BOLO export pipeline: fused .pth + embedded-optimized FP16 ONNX.

Final stage of the BOLO pipeline. Loads a trained checkpoint, fuses
Conv+BN, saves the raw state dict as .pth, exports FP32 ONNX (slimmed with
onnxslim), converts to FP16, and validates parity against PyTorch with
onnxruntime.

Usage:
    python -m bolo.export_bolo --weights runs/bolo/bolo_smoke/weights/best.pt

Outputs (project root):
    bolo11n_avian.pth        raw state dict of the fused model
    bolo11n_avian_fp32.onnx  slimmed FP32 graph (reference)
    bolo11n_avian_fp16.onnx  final embedded-optimized model
"""

import argparse
import time
from pathlib import Path

import torch

import bolo  # noqa: F401  (registers custom modules)
from bolo.model import BOLO_YAML

ROOT = Path(__file__).resolve().parent.parent
PTH_OUT = ROOT / "bolo11n_avian.pth"
FP32_OUT = ROOT / "bolo11n_avian_fp32.onnx"
FP16_OUT = ROOT / "bolo11n_avian_fp16.onnx"


def topologically_sort_graph(model):
    """Restore node order after ORT's FP16 converter adds input casts."""
    nodes = list(model.graph.node)
    available = {value.name for value in model.graph.input}
    available.update(value.name for value in model.graph.initializer)
    ordered = []
    while nodes:
        ready = [node for node in nodes if all(not name or name in available for name in node.input)]
        if not ready:
            raise RuntimeError("unable to topologically sort converted ONNX graph")
        for node in ready:
            nodes.remove(node)
            ordered.append(node)
            available.update(node.output)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    return model


def load_model(weights: str):
    """Rebuild BOLO from yaml and load trained weights (handles EMA ckpts)."""
    from bolo import build_bolo

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = None
    for key in ("ema", "model"):
        obj = ckpt.get(key)
        if obj is not None and hasattr(obj, "state_dict"):
            state = obj.state_dict()
            break
        if isinstance(obj, dict):
            state = obj
            break
    if state is None:
        raise ValueError(f"no model state found in {weights}")

    model = build_bolo(cfg=BOLO_YAML, verbose=False)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("criterion")}, strict=False)
    return model.eval().float()


def export_onnx(model, imgsz):
    import onnxslim

    dummy = torch.randn(1, 3, imgsz, imgsz)
    torch.onnx.export(
        model,
        dummy,
        str(FP32_OUT),
        # The legacy tracer is intentional: PyTorch 2.9's dynamo exporter
        # currently fails in YOLO Detect's cached anchor computation.
        dynamo=False,
        opset_version=18,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None,  # fixed shape for embedded deployment
    )
    slimmed = onnxslim.slim(str(FP32_OUT))
    import onnx

    onnx.save(slimmed, str(FP32_OUT))

    # ORT's converter preserves type information for Resize/cast graphs more
    # reliably than onnxconverter-common on current ONNX releases.
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model16 = convert_float_to_float16(onnx.load(str(FP32_OUT)), keep_io_types=True)
    model16 = topologically_sort_graph(model16)
    onnx.save(model16, str(FP16_OUT))
    return dummy


def validate(model, dummy):
    import onnx
    import onnxruntime as ort
    import numpy as np

    with torch.no_grad():
        ref = model(dummy)[0].numpy()
    for path, tag in ((FP32_OUT, "fp32"), (FP16_OUT, "fp16")):
        onnx.checker.check_model(str(path))
        sess = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        out = sess.run(None, {"images": dummy.numpy()})[0]
        diff = np.abs(out - ref).max()
        denom = np.abs(ref).max() + 1e-9
        print(f"[{tag}] output {out.shape}, max abs diff: {diff:.5f} (rel {diff / denom:.5%})")

        # latency on the active provider
        for _ in range(5):
            sess.run(None, {"images": dummy.numpy()})
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            sess.run(None, {"images": dummy.numpy()})
        print(f"[{tag}] latency: {(time.perf_counter() - t0) / reps * 1000:.1f} ms/img")


def main():
    ap = argparse.ArgumentParser(description="Export BOLO to .pth + FP16 ONNX")
    ap.add_argument("--weights", required=True, help="trained checkpoint (e.g. runs/bolo/.../weights/best.pt)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = load_model(args.weights)
    model.fuse()
    torch.save(model.state_dict(), PTH_OUT)
    n = sum(p.numel() for p in model.parameters())
    print(f"saved {PTH_OUT} ({n / 1e6:.2f}M params, {PTH_OUT.stat().st_size / 1e6:.1f} MB)")

    dummy = export_onnx(model, args.imgsz)
    print(f"saved {FP32_OUT} ({FP32_OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"saved {FP16_OUT} ({FP16_OUT.stat().st_size / 1e6:.1f} MB)")

    validate(model, dummy)


if __name__ == "__main__":
    main()
