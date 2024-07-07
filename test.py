# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

import benchmark.models.vae_keras as vae_keras
import benchmark.models.vae_torch as vae_torch
import benchmark.utils_keras as utils_keras
import benchmark.utils_onnx as utils_onnx
import benchmark.utils_sofie as utils_sofie
import benchmark.utils_torch as utils_torch

tf.config.experimental.set_visible_devices(devices=[], device_type="gpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# tf.debugging.set_log_device_placement(True)


# %%
plot_times = []
plot_ticks = []
plot_error = []

batch_size = 1
device = "cpu"

vae, x = vae_torch.get_vae_with_inputs(
    batch_size=batch_size,
    device=device,
)
time, times = utils_torch.time_model_inference(vae, x)
error = np.std(times) / np.sqrt(len(times))

plot_times.append(time)
plot_ticks.append(f"Torch {device}")
plot_error.append(error)

time, times = utils_onnx.time_model_inference(vae, x)
error = np.std(times) / np.sqrt(len(times))

plot_times.append(time)
plot_ticks.append(f"ONNX {device}")
plot_error.append(error)

vae, x = vae_keras.get_vae_with_inputs(
    batch_size=batch_size,
    device=device,
)
time, times = utils_keras.time_model_inference(vae, x)
error = np.std(times) / np.sqrt(len(times))

plot_times.append(time)
plot_ticks.append(f"Keras {device}")
plot_error.append(error)

print("SOFIE")
time, times = utils_sofie.time_model_inference(vae, x)
error = np.std(times) / np.sqrt(len(times))

plot_times.append(time)
plot_ticks.append(f"SOFIE {device}")
plot_error.append(error)


fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.bar(plot_ticks, plot_times)
ax.errorbar(
    list(range(len(plot_times))), plot_times, plot_error, ls="none", color="black"
)

plt.yscale("log")

ax.set_ylabel("Time [s]")
plt.title("Fastsim VAE Decoder Inference Times (batchsize 32)")
<<<<<<< HEAD
=======
plt.show()
>>>>>>> a8b2258 (update, some bug in SOFIE makes it run surprisingly fast)

# %%
