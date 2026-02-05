import drjit as dr
from drjit.auto.ad import TensorXf, Texture3f, Array3f, Float, Bool


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
        loss += dr.mean(dr.sqrt(dr.sum(dr.square(R_pred_z[1:] - R_target_z[1:]))))
    
    loss /= len(z_planes)
    return loss


def gradient(n_sq_tensor: TensorXf) -> Texture3f:
    # Central difference (f[i+1] - f[i-1]) / 2.0
    # This matches the precision requirements of the Runge-Kutta steps
    grad_z = (n_sq_tensor[2:, 1:-1, 1:-1] - n_sq_tensor[:-2, 1:-1, 1:-1])
    grad_y = (n_sq_tensor[1:-1, 2:, 1:-1] - n_sq_tensor[1:-1, :-2, 1:-1])
    grad_x = (n_sq_tensor[1:-1, 1:-1, 2:] - n_sq_tensor[1:-1, 1:-1, :-2])
    grad = 0.5 * dr.concat([grad_z, grad_y, grad_x], axis=-1)
    return Texture3f(grad)


def trace_rays_sharma(
    R_start: Array3f,
    T_start: Array3f,
    delta_t: float,
    grad_tex: Texture3f,
    cube_min: Array3f,
    cube_max: Array3f,
) -> tuple[Array3f, Array3f]:
    """
    Implements the Sharma et al. numerical ray tracing method.
    
    R_start: Array3f - Batch of initial ray positions.
    T_start: Array3f - Batch of initial optical ray vectors (n * dr/ds).
    delta_t: float   - The extrapolation distance (integration step).
    grad_tex: Texture3f - 3D texture containing n^2 values.
    cube_min/max: Array3f - Physical boundaries of the cubic medium.
    """
    
    # Initialize state from starting conditions
    R = Array3f(R_start)
    T = Array3f(T_start)
    active = Bool(True)

    def get_D(pos: Array3f) -> Array3f:
        """
        Computes D = 1/2 * grad(n^2).
        Uses autodiff to compute the spatial gradient through the texture.
        """
        # 1. Map physical position to [0, 1] for texture sampling
        uvw = (pos - cube_min) / (cube_max - cube_min)
        return Array3f(grad_tex.eval(uvw))

    # Runge-Kutta Integration Loop
    # This loop marches the rays until they exit the cube bounds
    max_steps = 1
    step = 0
    while step < max_steps:
        step += 1

        # TODO: enable later
        # if not active:
        #     break

        # Matrix A = delta_t * D(R_n)
        A = delta_t * get_D(R)
        
        # Matrix B = delta_t * D(R_n + 0.5*delta_t*T_n + 0.125*delta_t*A)
        R_b = R + 0.5 * delta_t * T + 0.125 * delta_t * A
        B = delta_t * get_D(R_b)
        
        # Matrix C = delta_t * D(R_n + delta_t*T_n + 0.5*delta_t*B)
        R_c = R + delta_t * T + 0.5 * delta_t * B
        C = delta_t * get_D(R_c)
        
        # Update Position R_{n+1}
        # R = R + delta_t * (T + (1.0/6.0) * (A + 2.0 * B))
        R[active] += delta_t * (T + (1.0/6.0) * (A + 2.0 * B))
         
        # Update Optical Ray Vector T_{n+1}
        T[active] += (1.0/6.0) * (A + 4.0 * B + C)
        
        # Termination: Ray exits the cubic medium
        within_bounds = dr.all((R >= cube_min) & (R <= cube_max))
        active &= within_bounds

    return R, T


def main():
    # --- Example Usage ---
    # Setup dummy texture data (e.g., a simple radial gradient)
    # n^2 = 2.5 - 0.1 * (x^2 + y^2)
    res = 512
    grid = dr.meshgrid(
        dr.linspace(Float, 0, 2, res), 
        dr.linspace(Float, -1, 1, res), 
        dr.linspace(Float, -1, 1, res)
    )
    n_data = 2.5 - 0.1 * (dr.square(grid[1]) + dr.square(grid[2]))
    n_data = dr.reshape(TensorXf, n_data, (res, res, res, 1))
    # n_data = Texture3f(n_data)
    dr.enable_grad(n_data)

    # Define Batch
    num_rays = 1024
    R0 = Array3f(
        0.0,
        dr.linspace(Float, -1, 1, num_rays),
        dr.linspace(Float, -1, 1, num_rays),
    )
    T0 = Array3f(1.0, 0.0, 0.0) # n * direction 

    n_epochs = 100
    lr = 0.01

    R, T = R0, T0

    for _ in range(n_epochs):
        # Trace
        n_grad = gradient(n_data)

        R, T = trace_rays_sharma(
            R, T, 0.1, n_grad, Array3f(0, -1, -1), Array3f(2, 1, 1)
        )

        loss = loss_func(R, T, R0, T0, (0.0, 1.0))
        print(f"loss: {loss.item()}")

        # Backpropagate
        dr.set_grad(n_data, 0.0)
        dr.backward(loss)

        # Update
        n_data = n_data - lr * dr.grad(n_data)
        dr.eval(n_data, R, T)  # reset variable tracing


if __name__ == "__main__":
    main()
