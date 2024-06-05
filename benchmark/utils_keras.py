from typing import Any, Callable, Literal, Mapping, Sequence

import performance
import tensorflow as tf


def time_model_inference(
    model: Callable,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
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

    # with PerformanceTestConfig(device=device):
    model = tf.function(model, jit_compile=True)

    synchronous = tf.config.experimental.get_synchronous_execution()
    
    r = performance.time_function_average(
        model,
        skip_first=True,
        args=args,
        kwargs=kwargs,
    )

    tf.config.experimental.set_synchronous_execution(synchronous)
    return r


def main():
    from models.vae_keras import get_vae

    vae = get_vae()

    with tf.device("cpu"):
        r = time_model_inference(
            vae, [tf.ones((32, 10))], {"training": False}
        )
        print(r[0])


if __name__ == "__main__":
    main()
