from typing import Any, Callable, Literal, Mapping, Sequence

import performance
import torch


def time_model_inference(
    model: Callable,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    *,
    device: Literal["cpu", "gpu"] | None = "cpu",
) -> tuple[float, list[float]]:
    if device == "gpu":
        device = "cuda"

    args = args or []
    kwargs = kwargs or {}

    model.eval()
    model.to(device)
    model = torch.jit.script(model, optimize=True)

    return performance.time_function_average(
        model,
        skip_first=True,
        args=args,
        kwargs=kwargs,
    )


def main():
    from models.vae_torch import get_vae

    vae = get_vae()
    r = time_model_inference(vae, [torch.ones((32, 10))], device="gpu")
    print(r[0])


if __name__ == "__main__":
    main()
