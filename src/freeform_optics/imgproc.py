"""Image processing utilities for drjit volume fields.

Functions operate on TensorXf fields of shape (nz, ny, nx, 1) and use
PyTorch as a backend for operations not available in drjit (e.g. convolution).
The drjit↔PyTorch conversion is zero-copy via the DLPack protocol.
"""

import torch
import torch.nn.functional as F
from drjit.auto.ad import TensorXf


def gaussian_filter(volume: TensorXf, sigma: float) -> TensorXf:
    """Smooth a volume field with a separable Gaussian kernel via PyTorch conv3d.

    Intended for smoothing gradient fields before applying an update step,
    to remove high-spatial-frequency content that isn't manufacturable.
    Should be called inside a `dr.suspend_grad()` block.

    Converts the drjit tensor to a PyTorch tensor zero-copy (via the DLPack
    protocol), applies three separable 1D conv3d passes (one per axis), then
    converts back.

    Parameters
    ----------
    volume : TensorXf
        Field of shape (nz, ny, nx, 1).
    sigma : float
        Standard deviation of the Gaussian kernel in voxels.
        sigma=0 returns the input unchanged.

    Returns
    -------
    TensorXf
        Smoothed field, same shape as input.

    Examples
    --------
    Typical usage inside a training loop::

        grad = dr.grad(optic.volume)
        with dr.suspend_grad():
            smooth_grad = smooth_volume(grad, sigma=2.0)
            new_volume = optic.volume - lr * smooth_grad
        optic = RefractiveOptic(optic.config, new_volume)
    """
    if sigma == 0.0:
        return volume

    # Zero-copy view as a PyTorch tensor (DLPack). Shape: (nz, ny, nx, 1).
    t = volume.torch()

    # conv3d expects (N, C, D, H, W). Squeeze the trailing channel, add batch
    # and channel dims: (nz, ny, nx, 1) → (1, 1, nz, ny, nx).
    a = t.squeeze(-1).unsqueeze(0).unsqueeze(0)

    # 1D Gaussian kernel, same approach as the reference smooth_3d().
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=t.dtype, device=t.device)
    kernel = torch.exp((-0.5 / sigma**2) * x**2)
    kernel = kernel / kernel.sum()
    klen = len(kernel)

    # Three separable passes — each touches one axis only.
    a = F.conv3d(a, kernel.view(1, 1, klen, 1, 1), padding="same")
    a = F.conv3d(a, kernel.view(1, 1, 1, klen, 1), padding="same")
    a = F.conv3d(a, kernel.view(1, 1, 1, 1, klen), padding="same")

    # Back to (nz, ny, nx, 1) and return as TensorXf.
    result = a.squeeze(0).squeeze(0).unsqueeze(-1).contiguous()
    return TensorXf(result)
