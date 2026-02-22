import math
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
from drjit.auto.ad import TensorXf, Texture3f, Array3f, Float, Bool, Array2i, Int, TensorXi


def z_propagate(
    R: Array3f,
    T: Array3f,
    z: float,
) -> Array3f:
    """
    Propagates the ray through the z-plane.
    """
    if z > 0:
        dt = (z - R[0]) / T[0]
        return R + dt * T
    else:
        return R


def loss_func(
    R_pred: Array3f,
    T_pred: Array3f,
    R_target: Array3f,
    T_target: Array3f,
    z_planes: tuple[float, ...],
) -> Float:

    loss = dr.zeros(Float)

    for z in z_planes:
        R_pred_z = z_propagate(R_pred, T_pred, z)
        R_target_z = z_propagate(R_target, T_target, z)
        loss += dr.mean(dr.norm(R_pred_z[1:] - R_target_z[1:]))
    
    loss /= len(z_planes)
    return loss


def gradient(n_sq_tensor: TensorXf) -> Texture3f:
    # Central difference (f[i+1] - f[i-1]) / 2.0
    # This matches the precision requirements of the Runge-Kutta steps
    grad_z = (n_sq_tensor[2:, 1:-1, 1:-1] - n_sq_tensor[:-2, 1:-1, 1:-1])
    grad_y = (n_sq_tensor[1:-1, 2:, 1:-1] - n_sq_tensor[1:-1, :-2, 1:-1])
    grad_x = (n_sq_tensor[1:-1, 1:-1, 2:] - n_sq_tensor[1:-1, 1:-1, :-2])
    plt.savefig("grad_z.png")
    grad = 0.5 * dr.concat([grad_z, grad_y, grad_x], axis=-1)
    return Texture3f(grad)


def trace_rays_sharma(
    R_start: Array3f,
    T_start: Array3f,
    dt: float,
    n_data: TensorXf,
    cube_min: Array3f,
    cube_max: Array3f,
) -> tuple[Array3f, Array3f]:
    """
    Implements the Sharma et al. numerical ray tracing method.
    
    R_start: Array3f - Batch of initial ray positions.
    T_start: Array3f - Batch of initial optical ray vectors (n * dr/ds).
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
        active: Bool, R: Array3f, T: Array3f,
    ) -> tuple[Array3f, Array3f, Bool]:

        # Matrix A = delta_t * D(R_n)
        A = dt * get_D(R, active)
        
        # Matrix B = delta_t * D(R_n + 0.5*delta_t*T_n + 0.125*delta_t*A)
        R_b = R + 0.5 * dt * T + 0.125 * dt * A
        B = dt * get_D(R_b, active)
        
        # Matrix C = delta_t * D(R_n + delta_t*T_n + 0.5*delta_t*B)
        R_c = R + dt * T + 0.5 * dt * B
        C = dt * get_D(R_c, active)
        
        # Update Position R_{n+1}
        R += dr.select(active, dt * (T + (1.0/6.0) * (A + 2.0 * B)), 0.0)
          
        # Update Optical Ray Vector T_{n+1}
        T += dr.select(active, (1.0/6.0) * (A + 4.0 * B + C), 0.0)

        within_bounds = dr.all(R >= cube_min) & dr.all(R <= cube_max)
        active = active & within_bounds

        return active, R, T

    state = (
        Bool(True),       # active
        Array3f(R_start), # R
        Array3f(T_start), # T
    )

    _, final_R, final_T = dr.while_loop(
        state,
        lambda active, R, T: active,
        _loop_body,
        label="trace_rays_sharma",
        max_iterations=-1,
    )

    return final_R, final_T


def render_rays(
    R: Array3f,
    cube_min: Array3f,
    cube_max: Array3f,
    shape: tuple[int, int],
) -> TensorXi:
    assert len(shape) == 2

    R_norm = ((R - cube_min) / (cube_max - cube_min))

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
    R_sq = 0.5 ** 2
    n_map = n0 / (1 + r_squared / R_sq)
    return n_map


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
) -> tuple[Array3f, Array3f]:
    # Point-picking on a sphere is easy, but also easy to do wrong:
    theta = np.arccos(np.random.uniform(np.cos(cone_angle/2), 1, num_rays))
    phi   =           np.random.uniform(0,              2*np.pi, num_rays)
    # Calculate our trig functions:
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sin_ph, cos_ph = np.sin(phi),   np.cos(phi)

    # Convert back to Cartesian:
    T0 = Array3f(np.stack([cos_th, sin_th * sin_ph, sin_th * cos_ph], axis=0))
    R0 = dr.zeros(Array3f, (3, num_rays))

    return R0, T0


def main():
    # --- Example Usage ---
    # Setup dummy texture data (e.g., a simple radial gradient)
    # n^2 = 2.5 - 0.1 * (x^2 + y^2)
    res = 512
    cube_min = (0, -1, -1)
    cube_max = (2, 1, 1)
    dr_cube_min = Array3f(cube_min)
    dr_cube_max = Array3f(cube_max)

    grid = dr.meshgrid(
        dr.linspace(Float, cube_min[0], cube_max[0], res), 
        dr.linspace(Float, cube_min[1], cube_max[1], res), 
        dr.linspace(Float, cube_min[2], cube_max[2], res)
    )

    ref_n_data = maxwell_fisheye(2.0, grid)
    ref_n_data = dr.reshape(TensorXf, ref_n_data, (res, res, res, 1))
    # ref_n_data = dr.full(TensorXf, 1.0, (res, res, res, 1))
    dr.enable_grad(ref_n_data)

    # Define Batch
    n_rays = 512 * 512
    # R0, T0 = uniform_rays(cube_min, cube_max, n_rays)
    R0, T0 = random_canonical_rays(np.pi/4, n_rays)
    # R0 += (1.0, 0.0, 0.0)

    R_target, _ = trace_rays_sharma(
        R0, T0, 0.001, ref_n_data, dr_cube_min, dr_cube_max
    )

    image_target = render_rays(R_target, dr_cube_min, dr_cube_max, (512, 512))
    # print(dr.whos_ad())
    image_target = image_target.numpy()
    print(image_target.shape, image_target.dtype)
    image_target = (image_target * 255).astype("uint8")
    from imageio import imwrite
    imwrite("image_target.png", image_target)
    return

    n_epochs = 100
    lr = 0.01

    R, T = R0, T0

    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(n_data.numpy().squeeze(), name="n_data")

    for _ in range(n_epochs):
        # Trace
        R, T = trace_rays_sharma(
            R, T, 0.1, n_data, Array3f(0, -1, -1), Array3f(2, 1, 1)
        )

        loss = loss_func(R, T, R0, T0, (0.0, 1.0))
        print(f"loss: {loss.item()}")

        # Backpropagate
        dr.set_grad(n_data, 0.0)
        dr.backward(loss)

        # Update
        n_data = n_data - lr * dr.grad(n_data)
        dr.eval(n_data, R, T)  # reset variable tracing
    
    # viewer.add_image(n_data.numpy().squeeze(), name="n_data_updated")
    # napari.run()


if __name__ == "__main__":
    main()
