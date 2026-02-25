import drjit as dr
from drjit.auto.ad import Array3f, Float


def z_propagate(
    zyx: Array3f,
    v_zyx: Array3f,
    z_offset: float,
    z_max: float,
) -> Array3f:
    """Propagate rays to the plane z = z_max + z_offset.

    Parameters
    ----------
    zyx : Array3f
        Current ray positions.
    v_zyx : Array3f
        Current ray velocities.
    z_offset : float
        Distance past z_max to propagate to. 0 means no propagation.
    z_max : float
        Exit z-coordinate of the optic domain (µm).

    Returns
    -------
    Array3f
        Ray positions at z = z_max + z_offset.
    """
    if z_offset == 0.0:
        return zyx
    dt = (z_max + z_offset - zyx[0]) / v_zyx[0]
    return zyx + dt * v_zyx


def imaging_loss(
    zyx_pred: Array3f,
    v_zyx_pred: Array3f,
    zyx_target: Array3f,
    v_zyx_target: Array3f,
    z_max: float,
    z_planes: tuple[float, ...],
) -> Float:
    """Mean ray distance in the YX plane at multiple z-planes past the exit.

    Measures the 2D (y, x) distance between predicted and target ray positions
    at each specified z-offset, averaged across rays and planes.

    Parameters
    ----------
    zyx_pred : Array3f
        Predicted exit ray positions.
    v_zyx_pred : Array3f
        Predicted exit ray velocities.
    zyx_target : Array3f
        Target exit ray positions.
    v_zyx_target : Array3f
        Target exit ray velocities.
    z_max : float
        Exit z-coordinate of the optic domain (µm).
    z_planes : tuple[float, ...]
        Z-offsets (µm) past z_max at which to evaluate the loss.

    Returns
    -------
    Float
        Scalar mean distance in µm.
    """
    loss = dr.zeros(Float)
    for z_offset in z_planes:
        pred = z_propagate(zyx_pred, v_zyx_pred, z_offset, z_max)
        target = z_propagate(zyx_target, v_zyx_target, z_offset, z_max)
        loss = loss + dr.mean(dr.norm(pred[1:] - target[1:]))
    return loss / len(z_planes)
