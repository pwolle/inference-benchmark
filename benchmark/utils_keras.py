from typing import Any, Mapping, Sequence

import keras
import tensorflow as tf
import numpy as np
import functools

from . import performance


def _get_device(model):
    return model.weights[0].device


def _synchronize_after(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        r = f(*args, **kwargs)
        tf.test.experimental.sync_devices()
        return r
    
    return wrapped


def time_model_inference(
    model: keras.Model,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, list[float]]:
    """
    Test the performance of a model by timing its execution. The model is
    wrapped in a tf.function to enable graph compilation.

    Parameters
    ---
    model : keras.Model
        The model to test. The model must be callable.

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

    synchronous = tf.config.experimental.get_synchronous_execution()
    tf.config.experimental.set_synchronous_execution(False)

    with tf.device(_get_device(model)):
        assert callable(model), "Model is not callable"

        model = tf.function(model, jit_compile=True)
        model = _synchronize_after(model)

        r = performance.time_function_average(
            model,
            skip_first=True,
            args=args,
            kwargs=kwargs,
        )

    tf.config.experimental.set_synchronous_execution(synchronous)
    return r


def main():
    from models.vae_keras import get_vae_with_inputs

    vae, args, kwargs = get_vae_with_inputs(batch_size=32, device="cpu")
    time, times = time_model_inference(vae, args=args, kwargs=kwargs)
    print(time, len(times))


if __name__ == "__main__":
    main()
