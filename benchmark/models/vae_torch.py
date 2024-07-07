from typing import Callable, Literal

import torch


def get_vae() -> Callable:
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 14),
        torch.nn.LeakyReLU(0.3),
        torch.nn.BatchNorm1d(14),
        torch.nn.Linear(14, 20),
        torch.nn.LeakyReLU(0.3),
        torch.nn.BatchNorm1d(20),
        torch.nn.Linear(20, 50),
        torch.nn.LeakyReLU(0.3),
        torch.nn.BatchNorm1d(50),
        torch.nn.Linear(50, 100),
        torch.nn.LeakyReLU(0.3),
        torch.nn.BatchNorm1d(100),
        torch.nn.Linear(100, 40500),
        torch.nn.Sigmoid(),
    )
    model.eval()
    return model


def get_vae_with_inputs(
    *,
    batch_size: int,
    device: Literal["cpu", "gpu", "cuda"],
) -> tuple[Callable, torch.Tensor]:
    if device == "gpu":
        device = "cuda"

    vae = get_vae().to(device)
    x = torch.randn((batch_size, 10)).to(device).detach().requires_grad_(False)
    return vae, x


def main() -> None:
    vae, x = get_vae_with_inputs(batch_size=32, device="cpu")
    print(vae(x).shape)


if __name__ == "__main__":
    main()
