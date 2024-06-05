import torch
import torch.nn as nn
import onnxruntime
import numpy as np


# Define simple Linear Regression PyTorch Model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# Set the model to inference mode and Export to ONNX format
input_dim = 3
output_dim = 1
model = LinearRegression(input_dim, output_dim).eval()
x = torch.randn(1, input_dim)
torch_out = model(x)

torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "model.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

# Load the ONNX model and Run the model using ONNX Runtime
ort_session = onnxruntime.InferenceSession("model.onnx")


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good !")
