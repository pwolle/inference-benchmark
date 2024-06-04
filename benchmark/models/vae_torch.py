import torch

from typing import Callable


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
        torch.nn.BatchNorm1d(14),
        torch.nn.Linear(100, 40500),
        torch.nn.Sigmoid(),
    )
    model.eval()
    return model