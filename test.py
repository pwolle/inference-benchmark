# %%
import benchmark.models.vae_keras as vae_keras
import benchmark.models.vae_torch as vae_torch
import benchmark.utils_keras as utils_keras
import benchmark.utils_torch as utils_torch

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# %%
plot_times = []
plot_ticks = []
plot_error = []

batch_size = 32
device = "cpu"

torch.set_num_threads(1)

vae, args, kwargs = vae_torch.get_vae_with_inputs(
    batch_size=batch_size,
    device=device,
)
time, times = utils_torch.time_model_inference(vae, args=args, kwargs=kwargs)
error = np.std(times) / np.sqrt(len(times)) 

plot_times.append(time)
plot_ticks.append(f"Torch {device}")
plot_error.append(error)


tf.config.threading.get_inter_op_parallelism_threads()
tf.config.threading.get_intra_op_parallelism_threads()

vae, args, kwargs = vae_keras.get_vae_with_inputs(
    batch_size=batch_size,
    device=device,
)
time, times = utils_keras.time_model_inference(vae, args=args, kwargs=kwargs)
error = np.std(times) / np.sqrt(len(times)) 

plot_times.append(time)
plot_ticks.append(f"Keras {device}")
plot_error.append(error)


fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.bar(plot_ticks, plot_times)
ax.errorbar(
    list(range(len(plot_times))),
    plot_times,
    plot_error,
    ls="none",
    color="black"
)

ax.set_ylabel("Time [s]")
plt.title("Fastsim VAE Decoder Inference Times (batchsize 32)", pad=5)

# %%
