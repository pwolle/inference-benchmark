# %%
import benchmark.models.vae_torch as vae_torch
import torch

batch_size = 1024

vae, x = vae_torch.get_vae_with_inputs(batch_size=batch_size, device="cpu")

path = "model.onnx"
onnx_program = torch.onnx.export(vae, x, path)
