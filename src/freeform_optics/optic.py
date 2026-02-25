from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
import numpy as np
from drjit.auto.ad import Array3f, Bool, TensorXf, Texture3f


@dataclass(frozen=True)
class RefractiveOpticConfig:
    """Configuration for a 3D gradient-index refractive optic.

    Physical units are microns. Coordinate ordering is (z, y, x) throughout,
    matching the drjit convention used in the Sharma ray tracer.
    """

    cube_min: tuple[float, float, float] = (0.0, -10.0, -10.0)  # (z, y, x) µm
    cube_max: tuple[float, float, float] = (20.0, +10.0, +10.0)  # (z, y, x) µm
    resolution: tuple[int, int, int] = (101, 101, 101)  # (nz, ny, nx)
    n_min: float = 1.4
    n_max: float = 1.6
    dt: float = 0.1  # Integration step (µm); ~voxel_size / 2


def _sigmoid(x: TensorXf) -> TensorXf:
    return 1.0 / (1.0 + dr.exp(-x))


def to_physical_concentration(config: RefractiveOpticConfig, volume: TensorXf) -> TensorXf:
    """Map an unconstrained volume field to refractive index n.

    Applies sigmoid to squash the unconstrained volume to (0, 1), then scales:
        n = n_min + (n_max - n_min) * sigmoid(volume)

    Using sigmoid instead of clip keeps the volume unconstrained during
    optimization — the gradient update is simply `volume -= lr * grad(volume)`
    with no clamping required.

    Parameters
    ----------
    config : RefractiveOpticConfig
        Provides n_min and n_max.
    volume : TensorXf
        Unconstrained field (logit space). Shape (nz, ny, nx, 1).
        sigmoid(0) = 0.5, i.e. volume=0 → midpoint concentration.

    Returns
    -------
    TensorXf
        Refractive index field n. Shape (nz, ny, nx, 1).
    """
    n = config.n_min + (config.n_max - config.n_min) * _sigmoid(volume)
    return n


def _gradient(n_sq: TensorXf) -> TensorXf:
    """Gradient of n² with air boundary conditions.

    Uses central differences in the interior and one-sided differences at each
    face, treating the implicit exterior as air (n_air² = 1.0). This avoids
    padding while correctly modelling the air-optic interface.

    Returns a plain TensorXf so the AD graph from n_sq to gradient values is
    preserved for when the caller wraps it in a Texture3f.

    Parameters
    ----------
    n_sq : TensorXf
        Squared refractive index field. Shape (nz, ny, nx, 1).

    Returns
    -------
    TensorXf
        Gradient field of shape (nz, ny, nx, 3), channels (grad_z, grad_y, grad_x).
    """
    n_air_sq = 1.0

    # Z axis: low face, interior, high face
    gz_low = (n_sq[1:2] - n_air_sq) / 2  # (1, ny, nx, 1)
    gz_mid = (n_sq[2:] - n_sq[:-2]) / 2  # (nz-2, ny, nx, 1)
    gz_high = (n_air_sq - n_sq[-2:-1]) / 2  # (1, ny, nx, 1)
    grad_z = dr.concat([gz_low, gz_mid, gz_high], axis=0)  # (nz, ny, nx, 1)

    # Y axis: low face, interior, high face
    gy_low = (n_sq[:, 1:2] - n_air_sq) / 2
    gy_mid = (n_sq[:, 2:] - n_sq[:, :-2]) / 2
    gy_high = (n_air_sq - n_sq[:, -2:-1]) / 2
    grad_y = dr.concat([gy_low, gy_mid, gy_high], axis=1)  # (nz, ny, nx, 1)

    # X axis: low face, interior, high face
    gx_low = (n_sq[:, :, 1:2] - n_air_sq) / 2
    gx_mid = (n_sq[:, :, 2:] - n_sq[:, :, :-2]) / 2
    gx_high = (n_air_sq - n_sq[:, :, -2:-1]) / 2
    grad_x = dr.concat([gx_low, gx_mid, gx_high], axis=2)  # (nz, ny, nx, 1)

    return dr.concat([grad_z, grad_y, grad_x], axis=-1)  # (nz, ny, nx, 3)


def _trace_rays_sharma(
    zyx_start: Array3f,
    v_zyx_start: Array3f,
    dt: float,
    n_grad_tex: Texture3f,
    cube_min: Array3f,
    cube_max: Array3f,
) -> tuple[Array3f, Array3f]:
    """Sharma et al. Runge-Kutta ray tracing through a gradient-index medium.

    Rays propagate until they exit the cube [cube_min, cube_max]. Each ray
    is deactivated independently when it leaves the domain.

    Input velocity magnitude must equal the local refractive index at the
    start position. For rays entering from air (n=1.0), unit direction
    vectors are correct.

    Gradient flow in symbolic dr.while_loop: Texture3f only propagates
    gradients through sample coordinates, not through texture data directly.
    The 3-evaluation RK scheme below relies on this: B_pos depends on A
    (first eval output) and C_pos depends on B (second eval output), so the
    UV-coordinate gradient of the second and third evals creates an indirect
    chain back through the outer AD tape to the original n² field.

    Parameters
    ----------
    zyx_start : Array3f
        Initial ray positions, shape (3, n_rays).
    v_zyx_start : Array3f
        Initial ray velocities (n * direction), shape (3, n_rays).
    dt : float
        Integration step in microns.
    n_grad_tex : Texture3f
        Gradient of n² wrapped in a Texture3f, built from _gradient() output.
        Must NOT have dr.enable_grad called on it directly — gradients must
        flow through the tensor it was built from, not the texture leaf.
    cube_min : Array3f
        Domain lower bound.
    cube_max : Array3f
        Domain upper bound.

    Returns
    -------
    tuple[Array3f, Array3f]
        (final_zyx, final_v_zyx): positions and velocities at exit.
    """

    def _get_acceleration(pos: Array3f, active: Bool) -> Array3f:
        uvw = (pos - cube_min) / (cube_max - cube_min)
        return Array3f(n_grad_tex.eval(uvw, active))

    def _loop_body(
        active: Bool,
        zyx: Array3f,
        v_zyx: Array3f,
    ) -> tuple[Bool, Array3f, Array3f]:
        A = dt * _get_acceleration(zyx, active)
        B = dt * _get_acceleration(zyx + 0.5 * dt * v_zyx + 0.125 * dt * A, active)
        C = dt * _get_acceleration(zyx + dt * v_zyx + 0.5 * dt * B, active)

        new_zyx = zyx + dr.select(active, dt * (v_zyx + (1.0 / 6.0) * (A + 2.0 * B)), 0.0)
        new_v_zyx = v_zyx + dr.select(active, (1.0 / 6.0) * (A + 4.0 * B + C), 0.0)

        within_bounds = dr.all(new_zyx >= cube_min) & dr.all(new_zyx <= cube_max)
        new_active = active & within_bounds

        return new_active, new_zyx, new_v_zyx

    _, final_zyx, final_v_zyx = dr.while_loop(
        (Bool(True), Array3f(zyx_start), Array3f(v_zyx_start)),
        lambda active, zyx, v_zyx: active,
        _loop_body,
        label="sharma_rk",
        max_iterations=-1,  # required for reverse-mode AD through the loop
    )
    return final_zyx, final_v_zyx


class RefractiveOptic:
    """A 3D gradient-index refractive optic.

    Holds a config and an unconstrained volume field (logit space). The mapping
    to physical refractive index is done by `to_physical_concentration` at trace
    time, which applies sigmoid to keep n within [n_min, n_max] without clamping.

    For gradient-based optimization:
        dr.enable_grad(optic.volume)
        zyx, v = optic.trace_rays(...)
        dr.backward(loss)
        new_volume = optic.volume - lr * dr.grad(optic.volume)  # no clip needed
        optic = RefractiveOptic(optic.config, new_volume)
    """

    def __init__(
        self,
        config: RefractiveOpticConfig,
        volume: TensorXf | None = None,
    ) -> None:
        """
        Parameters
        ----------
        config : RefractiveOpticConfig
            Frozen configuration (domain bounds, resolution, index range).
        volume : TensorXf, optional
            Unconstrained volume field (logit space). Shape (nz, ny, nx, 1).
            Defaults to zeros, which maps to midpoint concentration via sigmoid.
        """
        self.config = config
        nz, ny, nx = config.resolution
        if volume is None:
            self.volume = dr.zeros(TensorXf, (nz, ny, nx, 1))  # sigmoid(0) = 0.5
        else:
            self.volume = volume

    def trace_rays(
        self,
        zyx: Array3f,
        v_zyx: Array3f,
    ) -> tuple[Array3f, Array3f]:
        """Propagate rays through the optic via the Sharma RK method.

        Converts the volume field to n² via volume_to_n_sq, builds the
        gradient texture, then runs the RK loop until all rays exit the domain.

        Parameters
        ----------
        zyx : Array3f
            Initial ray positions, shape (3, n_rays). Rays should start at
            the entry face (z = cube_min[0]).
        v_zyx : Array3f
            Initial ray velocities (n * direction), shape (3, n_rays).
            For rays entering from air (n=1.0), pass unit direction vectors.

        Returns
        -------
        tuple[Array3f, Array3f]
            (final_zyx, final_v_zyx): exit positions and velocities.
        """
        n_sq = dr.square(to_physical_concentration(self.config, self.volume))
        n_grad = _gradient(n_sq)
        n_grad_tex = Texture3f(n_grad)  # do NOT call dr.enable_grad on this
        cube_min = Array3f(self.config.cube_min)
        cube_max = Array3f(self.config.cube_max)
        return _trace_rays_sharma(zyx, v_zyx, self.config.dt, n_grad_tex, cube_min, cube_max)

    @classmethod
    def from_maxwell_fisheye(
        cls,
        config: RefractiveOpticConfig,
        n0: float = 2.0,
    ) -> RefractiveOptic:
        """Construct a RefractiveOptic matching the Maxwell Fish-Eye index.

        n(r) = n0 / (1 + (r/R)²), where r is the distance from the domain
        centre, R = half the z-extent. Voxels where n < n_min (cube corners
        outside the sphere) are clamped to n_min concentration.

        Parameters
        ----------
        config : RefractiveOpticConfig
            Must have n_max >= n0; raises ValueError otherwise.
        n0 : float
            Peak refractive index at the sphere center. Default 2.0.
        """
        if n0 > config.n_max:
            raise ValueError(f"n0={n0} exceeds config.n_max={config.n_max}. Increase n_max.")

        nz, ny, nx = config.resolution
        z_min, y_min, x_min = config.cube_min
        z_max, y_max, x_max = config.cube_max

        z_center = (z_min + z_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        x_center = (x_min + x_max) / 2.0
        R = (z_max - z_min) / 2.0

        z = np.linspace(z_min, z_max, nz).reshape(nz, 1, 1)
        y = np.linspace(y_min, y_max, ny).reshape(1, ny, 1)
        x = np.linspace(x_min, x_max, nx).reshape(1, 1, nx)

        r_sq = (z - z_center) ** 2 + (y - y_center) ** 2 + (x - x_center) ** 2
        n = n0 / (1.0 + r_sq / R**2)

        c = np.clip((n - config.n_min) / (config.n_max - config.n_min), 1e-6, 1 - 1e-6)
        volume = np.log(c / (1.0 - c)).astype(np.float32)  # logit: inverse of sigmoid
        return cls(config, TensorXf(volume.reshape(nz, ny, nx, 1)))

    @classmethod
    def from_luneburg(cls, config: RefractiveOpticConfig) -> RefractiveOptic:
        """Construct a RefractiveOptic matching the Luneburg lens index.

        n(r) = sqrt(2 - (r/R)²) for r ≤ R, n = 1 outside,
        where R = half the z-extent and the lens is centered in the domain.

        Parameters
        ----------
        config : RefractiveOpticConfig
            Must have n_max >= sqrt(2) ≈ 1.414; raises ValueError otherwise.
        """
        n_peak = float(np.sqrt(2.0))
        if n_peak > config.n_max:
            raise ValueError(
                f"Luneburg peak n={n_peak:.3f} exceeds config.n_max={config.n_max}. Set n_max >= sqrt(2) ≈ 1.414."
            )

        nz, ny, nx = config.resolution
        z_min, y_min, x_min = config.cube_min
        z_max, y_max, x_max = config.cube_max

        z_center = (z_min + z_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        x_center = (x_min + x_max) / 2.0
        R = (z_max - z_min) / 2.0

        z = np.linspace(z_min, z_max, nz).reshape(nz, 1, 1)
        y = np.linspace(y_min, y_max, ny).reshape(1, ny, 1)
        x = np.linspace(x_min, x_max, nx).reshape(1, 1, nx)

        r_sq = (z - z_center) ** 2 + (y - y_center) ** 2 + (x - x_center) ** 2
        R_sq = R**2
        n = np.where(r_sq <= R_sq, np.sqrt(np.clip(2.0 - r_sq / R_sq, 0.0, None)), 1.0)

        c = np.clip((n - config.n_min) / (config.n_max - config.n_min), 1e-6, 1 - 1e-6)
        volume = np.log(c / (1.0 - c)).astype(np.float32)  # logit: inverse of sigmoid
        return cls(config, TensorXf(volume.reshape(nz, ny, nx, 1)))

    @classmethod
    def uniform(
        cls,
        config: RefractiveOpticConfig,
        value: float = 0.5,
    ) -> RefractiveOptic:
        """Construct a RefractiveOptic with a uniform volume field.

        This is the canonical starting point for gradient-based optimization.

        Parameters
        ----------
        config : RefractiveOpticConfig
            Frozen configuration (domain bounds, resolution, index range).
        value : float
            Physical concentration in [0, 1]. Default 0.5 (midpoint; stored as
            logit(0.5) = 0.0 in the unconstrained volume field).
        """
        nz, ny, nx = config.resolution
        c = float(np.clip(value, 1e-6, 1 - 1e-6))
        logit_value = float(np.log(c / (1.0 - c)))
        return cls(config, dr.full(TensorXf, logit_value, (nz, ny, nx, 1)))
