import drjit as dr
import numpy as np
from drjit.auto.ad import Array3f

from freeform_optics import RefractiveOptic, RefractiveOpticConfig, imaging_loss
from freeform_optics.rays import random_canonical_rays, uniform_rays


def _make_starting_rays(
    config: RefractiveOpticConfig,
    n_rays: int,
    rng: np.random.RandomState,
) -> tuple[Array3f, Array3f]:
    """Rays launched from the centre of the entry face into a forward cone."""
    return random_canonical_rays(
        cone_angle=np.pi / 6,
        num_rays=n_rays,
        cube_min=config.cube_min,
        cube_max=config.cube_max,
        rng=rng,
    )


def _train_step(
    optic: RefractiveOptic,
    zyx_0: Array3f,
    v_zyx_0: Array3f,
    zyx_target: Array3f,
    v_zyx_target: Array3f,
    lr: float,
    z_planes: tuple[float, ...],
) -> tuple[RefractiveOptic, float]:
    """Single gradient descent step. Returns updated optic and scalar loss."""
    volume = optic.volume
    dr.enable_grad(volume)

    zyx_pred, v_zyx_pred = RefractiveOptic(optic.config, volume).trace_rays(zyx_0, v_zyx_0)
    loss = imaging_loss(
        zyx_pred,
        v_zyx_pred,
        zyx_target,
        v_zyx_target,
        optic.config.cube_max[0],
        z_planes,
    )

    dr.set_grad(volume, 0.0)
    dr.backward(loss)

    loss_value = loss[0]
    grad = dr.grad(volume)
    with dr.suspend_grad():
        new_volume = volume - lr * grad  # no clip: sigmoid handles bounds

    return RefractiveOptic(optic.config, new_volume), loss_value


def test_maxwell_fisheye_convergence(training_config: RefractiveOpticConfig) -> None:
    """Optimizing from uniform volume converges toward Maxwell Fish-Eye behavior.

    Step 1: trace rays through the reference (Maxwell Fish-Eye) medium to get targets.
    Step 2: optimize a uniform optic against those targets.
    Step 3: assert the loss decreased by at least one order of magnitude.
    """
    rng = np.random.RandomState(42)
    n_rays = 256
    n_iterations = 300
    convergence_ratio = 0.1  # final loss must be < 10% of initial loss
    z_planes = (0.0, training_config.cube_max[0] * 0.05)

    # Step 1: generate target rays from the known reference medium
    reference = RefractiveOptic.from_maxwell_fisheye(training_config, n0=training_config.n_max)
    zyx_0, v_zyx_0 = _make_starting_rays(training_config, n_rays, rng)
    zyx_target, v_zyx_target = reference.trace_rays(zyx_0, v_zyx_0)

    # Step 2: optimize from uniform volume
    optic = RefractiveOptic.uniform(training_config)
    loss_initial = None
    loss_best = None

    for _ in range(n_iterations):
        optic, loss = _train_step(
            optic,
            zyx_0,
            v_zyx_0,
            zyx_target,
            v_zyx_target,
            lr=0.05,
            z_planes=z_planes,
        )
        if loss_initial is None:
            loss_initial = loss
        if loss_best is None or loss < loss_best:
            loss_best = loss

    # Step 3: assert convergence (track best, not final, to be robust to oscillation)
    assert loss_initial is not None and loss_best is not None
    assert loss_best < loss_initial * convergence_ratio, (
        f"Loss did not converge: initial={loss_initial:.4f}, best={loss_best:.4f}, "
        f"ratio={loss_best / loss_initial:.3f} (threshold={convergence_ratio})"
    )


def test_luneburg_convergence(training_config: RefractiveOpticConfig) -> None:
    """Optimizing from uniform volume converges toward Luneburg lens behavior."""
    # Luneburg peak is sqrt(2) ≈ 1.414, which is within n_max=2.0
    n_rays = 256
    n_iterations = 400
    convergence_ratio = 0.1
    z_planes = (0.0, training_config.cube_max[0] * 0.05)

    reference = RefractiveOptic.from_luneburg(training_config)
    zyx_0, v_zyx_0 = uniform_rays(training_config.cube_min, training_config.cube_max, n_rays)
    zyx_target, v_zyx_target = reference.trace_rays(zyx_0, v_zyx_0)

    optic = RefractiveOptic.uniform(training_config)
    loss_initial = None
    loss_best = None

    for _ in range(n_iterations):
        optic, loss = _train_step(
            optic,
            zyx_0,
            v_zyx_0,
            zyx_target,
            v_zyx_target,
            lr=0.05,
            z_planes=z_planes,
        )
        if loss_initial is None:
            loss_initial = loss
        if loss_best is None or loss < loss_best:
            loss_best = loss

    assert loss_initial is not None and loss_best is not None
    assert loss_best < loss_initial * convergence_ratio, (
        f"Loss did not converge: initial={loss_initial:.4f}, best={loss_best:.4f}, "
        f"ratio={loss_best / loss_initial:.3f} (threshold={convergence_ratio})"
    )
