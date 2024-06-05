import tensorflow as tf
import keras




def get_vae():
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
    model.build((1, 10,))
    return tf.function(lambda x: model(x, training=False), jit_compile=True)


vae = get_vae()

x = tf.ones((32, 10))
print(vae(x).shape)