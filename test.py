# %%
import benchmark.models.vae_keras as vae_keras
import benchmark.models.vae_torch as vae_torch
import benchmark.utils_keras as utils_keras
import benchmark.utils_torch as utils_torch

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

plot_times = []
plot_ticks = []
plot_error = []

for device in ["cpu", "gpu"]:
    batch_size = 32

    vae, args, kwargs = vae_keras.get_vae_with_inputs(
        batch_size=batch_size,
        device=device,
    )
    time, times = utils_keras.time_model_inference(vae, args=args, kwargs=kwargs)
    error = np.std(times) / np.sqrt(len(times)) 

    plot_times.append(time)
    plot_ticks.append(f"Keras {device}")
    plot_error.append(error)

    vae, args, kwargs = vae_torch.get_vae_with_inputs(
        batch_size=batch_size,
        device=device,
    )
    time, times = utils_torch.time_model_inference(vae, args=args, kwargs=kwargs)
    error = np.std(times) / np.sqrt(len(times)) 

    plot_times.append(time)
    plot_ticks.append(f"Torch {device}")
    plot_error.append(error)

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
