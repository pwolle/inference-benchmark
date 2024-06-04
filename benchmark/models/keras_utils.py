import tensorflow as tf
import keras
import numpy as np

from typing import Callable, Literal


tf.debugging.set_log_device_placement(True)


def get_keras_vae(
    *,
    batch_size: int = 32, 
    device: Literal["cpu", "gpu"] = "cpu",
) -> Callable:
    with tf.device(device):
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
        x = tf.random.normal((batch_size, 10,))
        model.build(x.shape)

        f = tf.function(lambda x: model(x, training=False), jit_compile=True)
        f(x)

        return lambda: np.array(f(x))


vae = get_keras_vae()
print(vae())