from typing import Any, Callable, Literal, Mapping, Self, Sequence

import performance
import tensorflow as tf


class PerformanceTestConfig:
    """
    Wrapper to set tensorflow configuration for performance testing. This
    includes setting the device and synchronous execution.
    """

    def __init__(
        self,
        *,
        device: Literal["cpu", "gpu"] | None = "cpu",
    ) -> None:
        self.device = device
        self.device_gpus = tf.config.get_visible_devices("GPU")
        self.synchronous = tf.config.experimental.get_synchronous_execution()

    def __enter__(self: Self):
        if self.device == "cpu":
            tf.config.set_visible_devices([], "GPU")

        tf.config.experimental.set_synchronous_execution(False)

    def __exit__(self: Self, *_) -> None:
        if self.device == "cpu":
            tf.config.set_visible_devices(self.device_gpus, "GPU")

        tf.config.experimental.set_synchronous_execution(self.synchronous)


def time_model_inference(
    model: Callable,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    *,
    device: Literal["cpu", "gpu"] | None = "cpu",
) -> tuple[float, list[float]]:
    """
    Test the performance of a model by timing its execution. The model is
    wrapped in a tf.function to enable graph compilation.

    Parameters
    ---
    model : Callable
        The model to test.

    args : Sequence[Any] | None
        The positional arguments to pass to the model, if None, an empty list
        is used.

    kwargs : Mapping[str, Any] | None
        The keyword arguments to pass to the model, if None, an empty
        dictionary is used.

    device : Literal["cpu", "gpu"] | None
        The device to run the model on, if None, "cpu" is used.

    Returns
    ---
    tuple[float, list[float]]
        The average time taken to execute the model in seconds and a list of
        the times taken to execute the model in seconds.
    """
    args = args or []
    kwargs = kwargs or {}

    with PerformanceTestConfig(device=device):
        model = tf.function(model, jit_compile=True)

        return performance.time_function_average(
            model,
            skip_first=True,
            args=args,
            kwargs=kwargs,
        )


def main():
    from models.vae_keras import get_vae

    vae = get_vae()
    r = time_model_inference(
        vae, [tf.ones((32, 10))], {"training": False}, device="cpu"
    )
    print(r[0])


if __name__ == "__main__":
    main()
