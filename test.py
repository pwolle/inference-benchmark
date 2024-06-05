import tensorflow as tf

from typing import Self, Literal


class KerasTestConfig:
    def __init__(
        self,
        device: Literal["cpu", "gpu"] | None = "cpu",
    ) -> None:
        self.device = device
        self.device_gpus = tf.config.get_visible_devices("GPU")
        self.synchronous = tf.config.experimental.get_synchronous_execution()

    def __enter__(self: Self):
        if self.device == "cpu":
            tf.config.set_visible_devices([], "GPU")

        tf.config.experimental.set_synchronous_execution(False)

    def __exit__(self: Self, *_):
        if self.device == "cpu":
            tf.config.set_visible_devices(self.device_gpus, "GPU")

        tf.config.experimental.set_synchronous_execution(self.synchronous)
