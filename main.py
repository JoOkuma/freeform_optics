import drjit as dr
import numpy as np
from drjit.auto.ad import TensorXf, Texture3f, Array3f, Array1f


def main():

    n_points_per_axis = 512
    shape = (521, 512, 512)

    tensor = dr.full(TensorXf, 0.0, (*shape, 1))
    dr.enable_grad(tensor)

    tex = Texture3f(tensor)

    pos = Array3f([
        i.ravel()
        for i in np.meshgrid(*[
            np.linspace(0, 1, n_points_per_axis)
            for _ in range(3)
        ], indexing="ij")]
    )

    values = Array1f(tex.eval(pos))

    dr.backward(values)

    grad = dr.grad(tensor)


if __name__ == "__main__":
    main()
