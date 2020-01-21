"""
Utilities module

Functions for plotting and debugging the SBO module.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

def set_random_seed(seed):
    r"""Sets the seed for generating random numbers

    Args:
        seed (int): The desired seed.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot2d_func(func, func_ranges, steps=100, levels=200):
    r"""Plots a contour plot of a 2D function

    Args:
        func (callable): a callable that specifies the 2D funtion
        func_ranges (list of lists): a list containing ranges for each dimension
            as a list of two values, [[x1_start, x1_end], [x2_start, x2_end]]
        steps (int): number of points to sample between
        levels (int): determines the number of the contour lines
    """

    x1_steps = torch.linspace(func_ranges[0][0], func_ranges[0][1], steps)
    x2_steps = torch.linspace(func_ranges[1][0], func_ranges[1][1], steps)

    x1_mesh, x2_mesh = torch.meshgrid(x1_steps, x2_steps)

    z_mesh = func(torch.stack((x1_mesh.flatten(), x2_mesh.flatten()),
                              dim=1)).reshape(steps, steps)

    plt.contourf(x1_mesh.detach().numpy(),
                 x2_mesh.detach().numpy(),
                 z_mesh.detach().numpy(), levels)

    plt.colorbar()
