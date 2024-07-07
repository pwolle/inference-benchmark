# %%
import torch
import onnx2torch


def convert(onnx_path, torch_path):
    model = onnx2torch.convert(onnx_path)
    model = torch.jit.script(model)
    torch.jit.save(model, torch_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "torch_path",
        type=str,
        help="Where to save the TorchScript model",
    )
    args = parser.parse_args()

    convert(args.onnx_path, args.torch_path)


if __name__ == "__main__":
    main()
