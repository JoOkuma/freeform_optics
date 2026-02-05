import drjit as dr
import numpy as np
from drjit.auto.ad import TensorXf, Texture3f, Array3f, Array1f, Float


def main():
    n_points_per_axis = 512
    shape = (521, 512, 512)

    rng = dr.rng(0)
    tensor = rng.uniform(TensorXf, (*shape, 3))
    # tensor = dr.full(TensorXf, 0.0, (*shape, 3))
    dr.enable_grad(tensor)

    tex = Texture3f(tensor)

    pos = Array3f(
        *[
            dr.ravel(i)
            for i in dr.meshgrid(
                dr.zeros(Float, 1),
                dr.linspace(Float, 0, 1, n_points_per_axis),
                dr.linspace(Float, 0, 1, n_points_per_axis),
            )

        ]
    )

    for _ in range(100):
        values = Array3f(tex.eval(pos))
        pos = pos + values

    dr.backward(values)
    grad = dr.grad(tensor)

    print(grad)


if __name__ == "__main__":
    main()
