"""Ray generation utilities for the Sharma ray tracer.

All functions use (z, y, x) coordinate ordering throughout, matching the
convention in `optic.py`. Rays entering from the left face of the domain
start at z = cube_min[0] and travel in the +z direction.

Velocity vectors represent `n * direction`, so a ray entering from air
(n = 1) with unit direction has |v| = 1.
"""

import math

import drjit as dr
import numpy as np
from drjit.auto.ad import Array3f, Float


def random_points_in_a_circle(
    num_rays: int,
    radius: float,
    cube_min: tuple[float, float, float],
    cube_max: tuple[float, float, float],
    rng: np.random.RandomState,
) -> Array3f:
    """Sample ray starting positions within a circle on the entry face of the domain.

    The circle is centred on the y-x centre of the domain and lies on the
    entry face (z = cube_min[0]). Radii are drawn from Uniform(0, radius),
    so the distribution is NOT area-uniform (it is biased toward the centre).
    Angles are drawn from Uniform(0, 2π).

    Parameters
    ----------
    num_rays : int
        Number of ray positions to generate.
    radius : float
        Radius of the circle in microns.
    cube_min : tuple[float, float, float]
        Domain lower bound (z, y, x) in microns. Sets z = cube_min[0] and
        the y-x centre from (cube_min + cube_max) / 2.
    cube_max : tuple[float, float, float]
        Domain upper bound (z, y, x) in microns.
    rng : np.random.RandomState
        Random number generator (pass an explicit seed for reproducibility).

    Returns
    -------
    Array3f
        Positions of shape (3, num_rays) in (z, y, x) order.
    """
    y_center = (cube_min[1] + cube_max[1]) / 2.0
    x_center = (cube_min[2] + cube_max[2]) / 2.0
    r = radius * rng.uniform(0, 1, num_rays)
    theta = rng.uniform(0, 2 * np.pi, num_rays)
    zyx = np.stack(
        [
            np.full(num_rays, cube_min[0]),
            y_center + r * np.sin(theta),
            x_center + r * np.cos(theta),
        ],
        axis=0,
    )
    return Array3f(zyx)


def random_directions_in_a_cone(
    num_rays: int,
    cone_angle: float,
    rng: np.random.RandomState,
) -> Array3f:
    """Sample unit direction vectors uniformly over a cone around the +z axis.

    Uses the standard solid-angle-uniform scheme: cos(θ) is drawn from
    Uniform(cos(cone_angle / 2), 1), which gives equal probability per unit
    solid angle. φ is drawn from Uniform(0, 2π).

    Parameters
    ----------
    num_rays : int
        Number of direction vectors to generate.
    cone_angle : float
        Full opening angle of the cone in radians (half-angle = cone_angle / 2).
        cone_angle = 0 → all rays along +z; cone_angle = π → full hemisphere.
    rng : np.random.RandomState
        Random number generator (pass an explicit seed for reproducibility).

    Returns
    -------
    Array3f
        Unit direction vectors of shape (3, num_rays) in (z, y, x) order.
    """
    theta = np.arccos(rng.uniform(np.cos(cone_angle / 2), 1, num_rays))
    phi = rng.uniform(0, 2 * np.pi, num_rays)
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sin_ph, cos_ph = np.sin(phi), np.cos(phi)
    return Array3f(np.stack([cos_th, sin_th * sin_ph, sin_th * cos_ph], axis=0))


def uniform_rays(
    cube_min: tuple[float, float, float],
    cube_max: tuple[float, float, float],
    num_rays: int,
) -> tuple[Array3f, Array3f]:
    """Create a regular grid of parallel rays entering the domain from the z = 0 face.

    Positions are arranged on an n x n grid spanning the full y-x extent of
    the domain, where n = floor(sqrt(num_rays)). All rays travel in the +z
    direction with speed 1 (air, n = 1).

    Designed for the Luneburg lens test: parallel rays entering the entry face
    should focus to a point on the exit face.

    Parameters
    ----------
    cube_min : tuple[float, float, float]
        Domain lower bound (z, y, x) in microns.
    cube_max : tuple[float, float, float]
        Domain upper bound (z, y, x) in microns.
    num_rays : int
        Requested number of rays. Actual count will be floor(sqrt(num_rays))².

    Returns
    -------
    tuple[Array3f, Array3f]
        (zyx, v_zyx): positions and velocities, each of shape (3, n²).
        Positions have z = cube_min[0] and y, x on the grid.
        Velocities are (1, 0, 0) — +z direction, unit speed.
    """
    n_rays_per_axis = int(math.sqrt(num_rays))
    zyx = Array3f(
        cube_min[0],
        *dr.meshgrid(
            dr.linspace(Float, cube_min[1], cube_max[1], n_rays_per_axis),
            dr.linspace(Float, cube_min[2], cube_max[2], n_rays_per_axis),
        ),
    )
    v_zyx = Array3f(1.0, 0.0, 0.0)  # n * direction
    return zyx, v_zyx


def random_canonical_rays(
    cone_angle: float,
    num_rays: int,
    cube_min: tuple[float, float, float],
    cube_max: tuple[float, float, float],
    rng: np.random.RandomState,
) -> tuple[Array3f, Array3f]:
    """Create rays starting at the centre of the entry face with random directions.

    All rays originate at the y-x centre of the domain on the entry face
    (z = cube_min[0]) and fan out within a cone around the +z axis. This is
    the canonical setup for the Maxwell Fish-Eye test: a point source on the
    sphere surface that the lens should image to the antipodal point.

    Parameters
    ----------
    cone_angle : float
        Full opening angle of the cone in radians (passed to
        `random_directions_in_a_cone`).
    num_rays : int
        Number of rays to generate.
    cube_min : tuple[float, float, float]
        Domain lower bound (z, y, x) in microns.
    cube_max : tuple[float, float, float]
        Domain upper bound (z, y, x) in microns.
    rng : np.random.RandomState
        Random number generator (pass an explicit seed for reproducibility).

    Returns
    -------
    tuple[Array3f, Array3f]
        (zyx, v_zyx): positions and velocities, each of shape (3, num_rays).
        All positions are at the domain centre of the entry face; velocities
        are unit vectors within the cone.
    """
    v_zyx = random_directions_in_a_cone(num_rays, cone_angle, rng)
    origin = np.array(
        [
            cube_min[0],
            (cube_min[1] + cube_max[1]) / 2.0,
            (cube_min[2] + cube_max[2]) / 2.0,
        ],
        dtype=np.float32,
    )
    zyx = Array3f(np.tile(origin[:, None], (1, num_rays)))
    return zyx, v_zyx
