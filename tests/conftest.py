import pytest

from freeform_optics import RefractiveOpticConfig


@pytest.fixture
def training_config() -> RefractiveOpticConfig:
    """Low-resolution config for fast test runs.

    Same physical dimensions as the reference software (µm).
    Resolution reduced from 101³ to 32³ and dt coarsened to match.
    n_max=2.0 to accommodate the Maxwell Fish-Eye peak index.
    """
    return RefractiveOpticConfig(
        cube_min=(0.0, -10.0, -10.0),
        cube_max=(20.0, +10.0, +10.0),
        resolution=(32, 32, 32),
        n_min=1.0,
        n_max=2.0,
        dt=0.5,
    )
