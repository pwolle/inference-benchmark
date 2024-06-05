import tempfile

import onnx
import onnxruntime
import torch
from models.vae_torch import get_vae_with_inputs


vae, (x,), _ = get_vae_with_inputs(batch_size=32, device="cpu")
print(x.shape)

with tempfile.NamedTemporaryFile() as f:
    torch.onnx.export(vae, x, f.name, verbose=False)

    onnx_model = onnx.load(f.name)
    onnx.checker.check_model(onnx_model)

    model = onnxruntime.InferenceSession(
        f.name,
        input_names=["input"],  # the model's input names
        output_names=["output"],
    )

    y = model.run(None, {"input": x.numpy()})
    print(y)
