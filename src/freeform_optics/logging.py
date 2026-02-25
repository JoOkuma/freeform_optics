from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from freeform_optics.optic import RefractiveOptic


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for training loop visualization and checkpointing.

    Parameters
    ----------
    output_dir : Path
        Directory to write images and loss plots.
    frequency : int
        Log every N iterations.
    overwrite : bool
        If True, overwrite the same file each time (useful for live monitoring).
        If False, append the iteration index to the filename so every snapshot
        is preserved (e.g., concentration_000100.png).
    """

    output_dir: Path
    frequency: int = 10
    overwrite: bool = True


class TrainingLogger:
    """Callback for logging training progress.

    Attach to a training loop via on_iteration(). All file I/O is isolated
    here; the training loop itself remains side-effect free.

    Parameters
    ----------
    config : LoggingConfig

    Examples
    --------
    >>> logger = TrainingLogger(LoggingConfig(output_dir=Path("output")))
    >>> for iteration in range(n_epochs):
    ...     optic, loss = train_step(optic, ...)
    ...     logger.on_iteration(iteration, optic, loss)
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        self._loss_history: list[float] = []
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def on_iteration(
        self,
        iteration: int,
        optic: RefractiveOptic,
        loss: float,
    ) -> None:
        """Called each iteration by the training loop.

        Saves a mid-plane concentration cross-section and a loss curve
        according to the configured frequency and overwrite settings.

        Parameters
        ----------
        iteration : int
            Current iteration index (0-based).
        optic : RefractiveOptic
            Current state of the optic being optimized.
        loss : float
            Loss value for this iteration.
        """
        self._loss_history.append(loss)

        if iteration % self._config.frequency != 0:
            return

        suffix = "" if self._config.overwrite else f"_{iteration:06d}"
        self._save_concentration_slice(optic, suffix)
        self._save_loss_curve(suffix)

    def _filepath(self, name: str, suffix: str, ext: str) -> Path:
        return self._config.output_dir / f"{name}{suffix}.{ext}"

    def _save_concentration_slice(self, optic: RefractiveOptic, suffix: str) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("agg")  # prevent memory leak from repeated plotting

        from freeform_optics.optic import to_physical_concentration

        nz = optic.config.resolution[0]
        n = to_physical_concentration(optic.config, optic.volume)
        mid_slice = n.numpy().squeeze()[nz // 2]  # (ny, nx) cross-section at mid-z

        fig, ax = plt.subplots()
        im = ax.imshow(mid_slice, vmin=optic.config.n_min, vmax=optic.config.n_max, origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax, label="Refractive index")
        ax.set_title("Refractive index (mid z-plane)")
        fig.savefig(self._filepath("refractive_index", suffix, "png"))
        plt.close(fig)

    def _save_loss_curve(self, suffix: str) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("agg")

        fig, ax = plt.subplots()
        ax.plot(self._loss_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (µm)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.savefig(self._filepath("loss", suffix, "png"))
        plt.close(fig)
