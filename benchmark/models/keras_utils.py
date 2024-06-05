import tensorflow as tf
import keras
import numpy as np

from typing import Callable, Literal

print(tf.__version__)

print(tf.config.get_visible_devices("GPU"))
tf.config.set_visible_devices([], "GPU")

tf.debugging.set_log_device_placement(True)
tf.config.experimental.set_synchronous_execution(True)

def get_keras_vae(
    *,
    batch_size: int = 32, 
    device: Literal["cpu", "gpu"] = "/cpu:0",
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
        model.build((batch_size, 10))

    def f(x):
        with tf.device('gpu'):
            return model(x, training=False)
 

    # f = lambda x: model(x, training=False)
    f = tf.function(f, jit_compile=True)
    x = tf.random.normal((batch_size, 10,), name="random")
    print(x.device)
    f(x)

    return lambda: np.array(f(x))


vae = get_keras_vae()
print(vae().shape)