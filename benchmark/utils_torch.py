import functools
from typing import Any, Callable

import torch

from . import performance


def _synchronize_after(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        r = f(*args, **kwargs)
        torch.cuda.synchronize()
        return r

    return wrapped


def time_model_inference(
    model: Callable[[torch.Tensor], Any],
    x: torch.Tensor,
) -> tuple[float, list[float]]:
    model.eval()

    model = torch.jit.script(model)
    model = _synchronize_after(model)

    return performance.time_function_average(
        model,
        skip_first=True,
        args=(x,),
    )


def main():
    from models.vae_torch import get_vae_with_inputs

    vae, x = get_vae_with_inputs(batch_size=32, device="cpu")
    time, times = time_model_inference(vae, x)
    print(time, len(times))


if __name__ == "__main__":
    main()
