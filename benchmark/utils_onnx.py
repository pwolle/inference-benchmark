import tempfile

import onnxruntime
import torch

from . import performance


def from_torch(model, inputs):
    with tempfile.NamedTemporaryFile() as f:
        torch.onnx.export(model, inputs, f.name, verbose=False)
        return onnxruntime.InferenceSession(f.name)
