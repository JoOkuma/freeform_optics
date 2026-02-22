import math
import numpy as np
import drjit as dr
from drjit.auto.ad import TensorXf, Texture3f, Array3f, Float, Bool, Array2i, Int, TensorXi


def random_points_in_a_circle(
    num_rays: int,
    radius: float,
    rng: np.random.RandomState,
) -> Array3f:
    r = radius * rng.uniform(0, 1, num_rays)
    theta = rng.uniform(0, 2 * np.pi, num_rays)
    zyx = np.stack(
        [np.zeros_like(theta), r * np.sin(theta), r * np.cos(theta)],
        axis=0,
    ) 
    return Array3f(zyx)


def random_directions_in_a_cone(
    num_rays: int,
    cone_angle: float,
    rng: np.random.RandomState,
) -> Array3f:
    theta = np.arccos(rng.uniform(np.cos(cone_angle/2), 1, num_rays))
    phi = rng.uniform(0, 2 * np.pi, num_rays)
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sin_ph, cos_ph = np.sin(phi),   np.cos(phi)
    return Array3f(np.stack([cos_th, sin_th * sin_ph, sin_th * cos_ph], axis=0))


def uniform_rays(
    cube_min: tuple[float, float, float],
    cube_max: tuple[float, float, float],
    num_rays: int,
) -> tuple[Array3f, Array3f]:
    n_rays_per_axis = int(math.sqrt(num_rays))
    R0 = Array3f(
        0.0,
        *dr.meshgrid(
            dr.linspace(Float, cube_min[1], cube_max[1], n_rays_per_axis),
            dr.linspace(Float, cube_min[2], cube_max[2], n_rays_per_axis),
        )
    )
    T0 = Array3f(1.0, 0.0, 0.0) # n * direction 
    return R0, T0


def random_canonical_rays(
    cone_angle: float,
    num_rays: int,
    rng: np.random.RandomState,
) -> tuple[Array3f, Array3f]:

    v_zyx = random_directions_in_a_cone(num_rays, cone_angle, rng)
    zyx = dr.zeros(Array3f, (3, num_rays))
    return zyx, v_zyx

