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


def main() -> None:
    vae = get_vae()

    x = tf.ones((32, 10))
    print(vae(x).shape)


if __name__ == "__main__":
    main()
