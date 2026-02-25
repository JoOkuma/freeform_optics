"""Demo: design a freeform 3D refractive optic via gradient descent.

Targets the Maxwell Fish-Eye input/output behaviour as a reference,
starting from a uniform medium and optimizing with drjit AD.

Run with:
    uv run python main.py
"""

from pathlib import Path

import drjit as dr
import numpy as np

from freeform_optics import (
    LoggingConfig,
    RefractiveOptic,
    RefractiveOpticConfig,
    TrainingLogger,
    imaging_loss,
)
from freeform_optics.rays import random_directions_in_a_cone, random_points_in_a_circle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def main() -> None:
    CONFIG = RefractiveOpticConfig(
        cube_min=(0.0, -10.0, -10.0),
        cube_max=(20.0, +10.0, +10.0),
        resolution=(64, 64, 64),
        n_min=1.4,
        n_max=1.6,
        dt=0.1,
    )

    N_RAYS = 512
    N_EPOCHS = 1000
    LR = 10
    Z_PLANES = (0.0, 1.0)

    rng = np.random.RandomState(42)

    # Reference medium and fixed ray bundle
    reference = RefractiveOptic.from_maxwell_fisheye(CONFIG, n0=CONFIG.n_max)

    # Optimization loop
    optic = RefractiveOptic.uniform(CONFIG)
    logger = TrainingLogger(LoggingConfig(output_dir=Path("output"), frequency=50))

    for iteration in range(N_EPOCHS):
        # Sample random ray bundle
        zyx_0 = random_points_in_a_circle(
            N_RAYS, radius=1.0, cube_min=CONFIG.cube_min, cube_max=CONFIG.cube_max, rng=rng
        )
        v_zyx_0 = random_directions_in_a_cone(N_RAYS, cone_angle=np.pi / 6, rng=rng)

        zyx_target, v_zyx_target = reference.trace_rays(zyx_0, v_zyx_0)

        dr.enable_grad(optic.volume)
        zyx_pred, v_zyx_pred = optic.trace_rays(zyx_0, v_zyx_0)

        loss = imaging_loss(
            zyx_pred,
            v_zyx_pred,
            zyx_target,
            v_zyx_target,
            CONFIG.cube_max[0],
            Z_PLANES,
        )

        dr.set_grad(optic.volume, 0.0)
        dr.backward(loss)

        loss_value = loss[0]
        grad = dr.grad(optic.volume)
        with dr.suspend_grad():
            optic.volume = optic.volume - LR * grad

        print(f"iteration {iteration:04d}  loss={loss_value:.4f} µm")
        logger.on_iteration(iteration, optic, loss_value)


if __name__ == "__main__":
    main()
