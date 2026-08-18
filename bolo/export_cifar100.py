"""Export a trained BOLO-CIFAR100 checkpoint to valid FP32 and FP16 ONNX."""

import argparse
from pathlib import Path

import onnx
import torch

from bolo.cifar100_model import BoloCIFAR100
from bolo.export_bolo import topologically_sort_graph

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Export BOLO-CIFAR100")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-prefix", default=str(ROOT / "bolo_cifar100"))
    args = parser.parse_args()
    prefix = Path(args.output_prefix)
    pth, fp32, fp16 = prefix.with_suffix(".pth"), prefix.with_name(prefix.name + "_fp32.onnx"), prefix.with_name(prefix.name + "_fp16.onnx")

    model = BoloCIFAR100().eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    torch.save(model.state_dict(), pth)
    sample = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, sample, fp32, input_names=["images"], output_names=["logits"], opset_version=18, dynamo=False)
    import onnxslim
    onnx.save(onnxslim.slim(str(fp32)), fp32)
    from onnxruntime.transformers.float16 import convert_float_to_float16
    onnx.save(topologically_sort_graph(convert_float_to_float16(onnx.load(fp32), keep_io_types=True)), fp16)
    for path in (fp32, fp16):
        onnx.checker.check_model(str(path))
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
