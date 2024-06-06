from typing import Literal

import keras
import tensorflow as tf


def get_vae() -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.Dense(14, bias_initializer="random_normal"),
            keras.layers.LeakyReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(20, bias_initializer="random_normal"),
            keras.layers.LeakyReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(50, bias_initializer="random_normal"),
            keras.layers.LeakyReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(100, bias_initializer="random_normal"),
            keras.layers.LeakyReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(40500, bias_initializer="random_normal"),
            keras.layers.Activation("sigmoid"),
        ]
    )
    model.build((1, 10))
    return model


def get_vae_with_inputs(
    *,
    batch_size: int,
    device: Literal["cpu", "gpu", "cuda"],
) -> tuple[keras.Sequential, tf.Tensor]:
    if device == "cuda":
        device = "gpu"

    with tf.device(device):
        vae = get_vae()
        x = tf.random.normal((batch_size, 10))

    return vae, x


def main() -> None:
    vae, args, kwargs = get_vae_with_inputs(batch_size=32, device="cpu")
    print(vae(*args, **kwargs).shape)


if __name__ == "__main__":
    main()
