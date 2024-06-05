# %%
import benchmark.models.vae_keras as vae_keras
import benchmark.models.vae_torch as vae_torch
import benchmark.utils_keras as utils_keras
import benchmark.utils_torch as utils_torch

import numpy as np
import matplotlib.pyplot as plt


# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

plot_times = []
plot_ticks = []


device = "cpu"
batch_size = 32

vae, args, kwargs = vae_keras.get_vae_with_inputs(
    batch_size=batch_size,
    device=device,
)
time, times = utils_keras.time_model_inference(vae, args=args, kwargs=kwargs)

plot_times.append(time)
plot_ticks.append(f"Keras {device}")

vae, args, kwargs = vae_torch.get_vae_with_inputs(
    batch_size=batch_size,
    device=device,
)
time, times = utils_torch.time_model_inference(vae, args=args, kwargs=kwargs)

plot_times.append(time)
plot_ticks.append(f"Torch {device}")

plt.bar(plot_ticks, plot_times)
plt.ylabel("Time [s]")

# rotate x-ticks
plt.xticks(rotation=90)


# %%
