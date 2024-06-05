from typing import Any, Callable, Literal, Mapping, Sequence

import performance
import torch


def time_model_inference(
    model: Callable,
    *,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, list[float]]:
    args = args or []
    kwargs = kwargs or {}

    model.eval()
    model = torch.jit.script(model)

    return performance.time_function_average(
        model,
        skip_first=True,
        args=args,
        kwargs=kwargs,
    )


def main():
    from models.vae_torch import get_vae_with_inputs

    vae, args, kwargs = get_vae_with_inputs(batch_size=32, device="cpu")
    time, times = time_model_inference(vae, args=args, kwargs=kwargs)
    print(time, len(times))


if __name__ == "__main__":
    main()
