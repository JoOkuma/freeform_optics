import drjit as dr
import numpy as np
from drjit.auto.ad import TensorXf, Texture3f, Array3f, Array1f, Float, Bool


def trace_rays_sharma(
    R_start: Array3f,
    T_start: Array3f,
    delta_t: float,
    index_sq_texture: Texture3f,
    cube_min: Array3f,
    cube_max: Array3f,
) -> tuple[Array3f, Array3f]:
    """
    Implements the Sharma et al. numerical ray tracing method[cite: 1, 13].
    
    R_start: Array3f - Batch of initial ray positions[cite: 70].
    T_start: Array3f - Batch of initial optical ray vectors (n * dr/ds)[cite: 40, 70].
    delta_t: float   - The extrapolation distance (integration step)[cite: 77].
    index_sq_texture: Texture3f - 3D texture containing n^2 values[cite: 38, 50].
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
        
        # 2. Enable gradient tracking on the input position
        dr.enable_grad(uvw)
        
        # 3. Sample n^2 (refractive index squared) from texture [cite: 38]
        # Hardware interpolation provides a continuous field for the gradient
        n_sq = Array1f(index_sq_texture.eval(uvw))
        
        # 4. Backpropagate to find grad(n^2) [cite: 34]
        dr.backward(n_sq)
        grad_n_sq = dr.grad(uvw)
        
        # 5. Return D = 0.5 * grad(n^2) 
        return 0.5 * grad_n_sq

    # Runge-Kutta Integration Loop [cite: 14, 71]
    # This loop marches the rays until they exit the cube bounds
    max_steps = 100
    step = 0
    while step < max_steps:
        step += 1

        # TODO: enable later
        # if not active:
        #     break

        # Matrix A = delta_t * D(R_n) [cite: 74]
        A = delta_t * get_D(R)
        
        # Matrix B = delta_t * D(R_n + 0.5*delta_t*T_n + 0.125*delta_t*A) [cite: 75]
        R_b = R + 0.5 * delta_t * T + 0.125 * delta_t * A
        B = delta_t * get_D(R_b)
        
        # Matrix C = delta_t * D(R_n + delta_t*T_n + 0.5*delta_t*B) [cite: 75]
        R_c = R + delta_t * T + 0.5 * delta_t * B
        C = delta_t * get_D(R_c)
        
        # Update Position R_{n+1} [cite: 72]
        R[active] += delta_t * (T + (1.0/6.0) * (A + 2.0 * B))
        
        # Update Optical Ray Vector T_{n+1} [cite: 72]
        T[active] += (1.0/6.0) * (A + 4.0 * B + C)
        
        # Termination: Ray exits the cubic medium
        within_bounds = dr.all((R >= cube_min) & (R <= cube_max))
        active &= within_bounds

    return R, T


def main():
    # --- Example Usage ---
    # Setup dummy texture data (e.g., a simple radial gradient)
    # n^2 = 2.5 - 0.1 * (x^2 + y^2) [cite: 87]
    res = 512
    grid = dr.meshgrid(
        dr.linspace(Float, 0, 2, res), 
        dr.linspace(Float, -1, 1, res), 
        dr.linspace(Float, -1, 1, res)
    )
    n_sq_data = 2.5 - 0.1 * (dr.square(grid[1]) + dr.square(grid[2]))
    n_sq_data = dr.reshape(TensorXf, n_sq_data, (res, res, res, 1))

    tex = Texture3f(n_sq_data)
    dr.enable_grad(tex)

    # Define Batch
    num_rays = 1024
    R0 = Array3f(
        0.0,
        dr.linspace(Float, -1, 1, num_rays),
        dr.linspace(Float, -1, 1, num_rays),
    )
    T0 = Array3f(1.0, 0.0, 0.0) # n * direction 

    # Trace
    final_R, final_T = trace_rays_sharma(
        R0, T0, 0.1, tex, Array3f(0, -1, -1), Array3f(2, 1, 1)
    )
    print(final_R.shape, final_T.shape)


if __name__ == "__main__":
    main()
