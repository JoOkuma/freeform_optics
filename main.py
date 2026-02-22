import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
from drjit.auto.ad import (
    Array2i,
    Array3f,
    Bool,
    Float,
    TensorXf,
    TensorXi,
    Texture3f,
)

from _rays import (
    random_directions_in_a_cone,
    random_points_in_a_circle,
)


def z_propagate(
    zyx: Array3f,
    v_zyx: Array3f,
    z: float,
    z_max: float,
) -> Array3f:
    """
    Propagates the ray through the z-plane.
    """
    if z > 0:
        dt = (z - (zyx[0] - z_max)) / v_zyx[0]
        return zyx + dt * v_zyx
    else:
        return zyx


def loss_func(
    zyx_pred: Array3f,
    v_zyx_pred: Array3f,
    zyx_target: Array3f,
    v_zyx_target: Array3f,
    cube_max: tuple[float, float, float],
    z_planes: tuple[float, ...],
) -> Float:
    loss = dr.zeros(Float)
    for z in z_planes:
        pred_propagated = z_propagate(zyx_pred, v_zyx_pred, z, cube_max[0])
        target_propagated = z_propagate(zyx_target, v_zyx_target, z, cube_max[0])
        loss += dr.mean(dr.norm(pred_propagated[1:] - target_propagated[1:]))
    loss /= len(z_planes)
    return loss


def gradient(n_sq_tensor: TensorXf) -> Texture3f:
    # Central difference (f[i+1] - f[i-1]) / 2.0
    # This matches the precision requirements of the Runge-Kutta steps
    grad_z = n_sq_tensor[2:, 1:-1, 1:-1] - n_sq_tensor[:-2, 1:-1, 1:-1]
    grad_y = n_sq_tensor[1:-1, 2:, 1:-1] - n_sq_tensor[1:-1, :-2, 1:-1]
    grad_x = n_sq_tensor[1:-1, 1:-1, 2:] - n_sq_tensor[1:-1, 1:-1, :-2]
    plt.savefig("grad_z.png")
    grad = 0.5 * dr.concat([grad_z, grad_y, grad_x], axis=-1)
    return Texture3f(grad)


def trace_rays_sharma(
    zyx_start: Array3f,
    v_zyx_start: Array3f,
    dt: float,
    n_data: TensorXf,
    cube_min: Array3f,
    cube_max: Array3f,
) -> tuple[Array3f, Array3f]:
    """
    Implements the Sharma et al. numerical ray tracing method.

    zyx_start: Array3f - Batch of initial ray positions.
    v_zyx_start: Array3f - Batch of initial optical ray vectors (n * dr/ds).
    delta_t: float   - The extrapolation distance (integration step).
    n_data: TensorXf - 3D texture containing n^2 values.
    cube_min/max: Array3f - Physical boundaries of the cubic medium.
    """

    # Initialize state from starting conditions
    n_grad = gradient(n_data)

    def get_D(pos: Array3f, active: Bool) -> Array3f:
        """
        Computes D = 1/2 * grad(n^2).
        Uses autodiff to compute the spatial gradient through the texture.
        """
        # 1. Map physical position to [0, 1] for texture sampling
        uvw = (pos - cube_min) / (cube_max - cube_min)
        return Array3f(n_grad.eval(uvw, active))

    def _loop_body(
        active: Bool,
        zyx: Array3f,
        v_zyx: Array3f,
    ) -> tuple[Array3f, Array3f, Bool]:
        # Matrix A = delta_t * D(R_n)
        A = dt * get_D(zyx, active)

        # Matrix B = delta_t * D(R_n + 0.5*delta_t*T_n + 0.125*delta_t*A)
        B_zyx = zyx + 0.5 * dt * v_zyx + 0.125 * dt * A
        B = dt * get_D(B_zyx, active)

        # Matrix C = delta_t * D(R_n + delta_t*T_n + 0.5*delta_t*B)
        C_zyx = zyx + dt * v_zyx + 0.5 * dt * B
        C = dt * get_D(C_zyx, active)

        # Update Position R_{n+1}
        zyx += dr.select(active, dt * (v_zyx + (1.0 / 6.0) * (A + 2.0 * B)), 0.0)

        # Update Optical Ray Vector T_{n+1}
        v_zyx += dr.select(active, (1.0 / 6.0) * (A + 4.0 * B + C), 0.0)

        within_bounds = dr.all(zyx >= cube_min) & dr.all(zyx <= cube_max)
        active = active & within_bounds

        return active, zyx, v_zyx

    state = (
        Bool(True),  # active
        Array3f(zyx_start),  # R
        Array3f(v_zyx_start),  # T
    )

    _, final_R, final_T = dr.while_loop(
        state,
        lambda active, zyx, v_zyx: active,
        _loop_body,
        label="trace_rays_sharma",
        max_iterations=-1,
    )

    return final_R, final_T


def render_rays(
    zyx: Array3f,
    cube_min: Array3f,
    cube_max: Array3f,
    shape: tuple[int, int],
) -> TensorXi:
    assert len(shape) == 2

    R_norm = (zyx - cube_min) / (cube_max - cube_min)

    uv = Array2i(dr.round(shape * R_norm[1:] + 0.5))

    valid = R_norm[0] > 1.0  # reached the surface
    values = dr.select(valid, 1.0, 0.0)

    valid_np = valid.numpy()
    uv_np = uv.numpy()
    valid_uv_np = uv_np[:, valid_np]
    plt.hist2d(valid_uv_np[0], valid_uv_np[1])
    plt.savefig("hist.png")

    image = dr.zeros(Float, shape[0] * shape[1])
    dr.scatter_reduce(dr.ReduceOp.Add, image, values, uv[0] * shape[1] + uv[1])
    image = image / dr.max(image)

    return dr.reshape(TensorXf, image, (shape[0], shape[1]))


def maxwell_fisheye(
    n0: float,
    grid: tuple[TensorXf, TensorXf, TensorXf],
) -> TensorXf:
    # assuming z in (0, 2), x and y in (-1, 1)
    r_squared = dr.square(grid[0] - 1 + 1.0 / 512) + dr.square(grid[1]) + dr.square(grid[2])
    R_sq = 0.5**2
    n_map = n0 / (1 + r_squared / R_sq)
    return n_map


def main() -> None:
    res = 512
    rng = np.random.RandomState(42)
    cube_min = (0, -1, -1)
    cube_max = (2, 1, 1)
    dr_cube_min = Array3f(cube_min)
    dr_cube_max = Array3f(cube_max)

    grid = dr.meshgrid(
        dr.linspace(Float, cube_min[0], cube_max[0], res),
        dr.linspace(Float, cube_min[1], cube_max[1], res),
        dr.linspace(Float, cube_min[2], cube_max[2], res),
    )

    ref_n_data = maxwell_fisheye(2.0, grid)
    ref_n_data = dr.reshape(TensorXf, ref_n_data, (res, res, res, 1))
    # ref_n_data = dr.full(TensorXf, 1.0, (res, res, res, 1))

    # Define Batch
    n_rays = 512 * 512
    # zyx_0, v_zyx_0 = uniform_rays(cube_min, cube_max, n_rays)
    # zyx_0, v_zyx_0 = random_canonical_rays(np.pi/4, n_rays, rng)
    zyx_0 = random_points_in_a_circle(n_rays, 1.0, rng)
    v_zyx_0 = random_directions_in_a_cone(n_rays, np.pi / 4, rng)
    # zyx_0 += (1.0, 0.0, 0.0)

    zyx_target, zyx_target = trace_rays_sharma(zyx_0, v_zyx_0, 0.001, ref_n_data, dr_cube_min, dr_cube_max)

    image_target = render_rays(zyx_target, dr_cube_min, dr_cube_max, (512, 512))
    # print(dr.whos_ad())
    image_target = image_target.numpy()
    print(image_target.shape, image_target.dtype)
    image_target = (image_target * 255).astype("uint8")
    from imageio import imwrite

    imwrite("image_target.png", image_target)

    n_epochs = 100
    lr = 100

    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(n_data.numpy().squeeze(), name="n_data")
    n_data = dr.full(TensorXf, 1.5, (res, res, res, 1))
    dr.enable_grad(n_data)
    print(n_data.shape)

    plt.imshow(n_data.numpy().squeeze()[:, res // 2])
    plt.colorbar()
    plt.savefig("n_data.png")
    plt.close()

    n_rays = 1024

    for _ in range(n_epochs):
        zyx_0 = random_points_in_a_circle(n_rays, 0.5, rng)
        v_zyx_0 = random_directions_in_a_cone(n_rays, np.pi / 4, rng)

        zyx_target, zyx_target = trace_rays_sharma(zyx_0, v_zyx_0, 1 / res, ref_n_data, dr_cube_min, dr_cube_max)

        # Trace
        zyx, v_zyx = trace_rays_sharma(zyx_0, v_zyx_0, 1 / res, n_data, dr_cube_min, dr_cube_max)

        loss = loss_func(zyx, v_zyx, zyx_target, zyx_target, cube_max, (0.0, 0.05))
        print(f"loss: {loss.item()}")
        if loss.item() < 1e-6:
            print("loss is too small, stopping")
            break

        # Backpropagate
        dr.set_grad(n_data, 0.0)
        dr.backward(loss)

        plt.imshow(n_data.numpy().squeeze()[:, res // 2])
        plt.colorbar()
        plt.savefig("n_data.png")
        plt.close()

        # Update
        n_data_grad = dr.grad(n_data)
        with dr.suspend_grad(n_data):
            n_data = n_data - lr * n_data_grad
        dr.enable_grad(n_data)

    # viewer.add_image(n_data.numpy().squeeze(), name="n_data_updated")
    # napari.run()


if __name__ == "__main__":
    main()
