import tempfile

import onnx
import onnxruntime
import torch
from models.vae_torch import get_vae_with_inputs


vae, (x,), _ = get_vae_with_inputs(batch_size=32, device="cpu")
print(x.shape)

with tempfile.NamedTemporaryFile() as f:
    torch.onnx.export(
        vae,
        x,
        f.name,
        # verbose=False,
        input_names=["input"],  # the model's input names
        output_names=["output"],
    )

    onnx_model = onnx.load(f.name)
    onnx.checker.check_model(onnx_model)

    model = onnxruntime.InferenceSession(f.name)

    y = model.run(None, {"input": x.detach().cpu().numpy()})
    print(y)


# torch.onnx.export(
#     vae,  # model being run
#     x,  # model input (or a tuple for multiple inputs)
#     "model.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=True,  # store the trained parameter weights inside the model file
#     opset_version=10,  # the ONNX version to export the model to
#     do_constant_folding=True,  # whether to execute constant folding for optimization
#     input_names=["input"],  # the model's input names
#     output_names=["output"],  # the model's output names
#     dynamic_axes={
#         "input": {0: "batch_size"},  # variable length axes
#         "output": {0: "batch_size"},
#     },
# )


# ort_session = onnxruntime.InferenceSession("model.onnx")


# def to_numpy(tensor):
#     return (
#         tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#     )


# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)
