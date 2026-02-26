from __future__ import annotations

import math


def cosine_annealing_lr(
    step: int,
    lr_min: float,
    lr_max: float,
    n_steps: int,
) -> float:
    """Cosine annealing learning rate schedule.

    Decays the learning rate from lr_max to lr_min following a half-cosine
    curve over n_steps iterations, then holds at lr_min:

        lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / n_steps))

    Parameters
    ----------
    step : int
        Current iteration index (0-based).
    lr_min : float
        Floor learning rate.
    lr_max : float
        Peak learning rate, returned at step 0.
    n_steps : int
        Number of steps over which to decay from lr_max to lr_min.

    Examples
    --------
    >>> for step in range(n_epochs):
    ...     lr = cosine_annealing_lr(step, lr_max=10.0, n_steps=n_epochs, lr_min=0.1)
    ...     volume = volume - lr * grad
    """
    if lr_min >= lr_max:
        raise ValueError(f"'lr_min' must be less than 'lr_max': {lr_min} >= {lr_max}")
    t = min(step, n_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / n_steps))
