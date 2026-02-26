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
        Directory to write images and metric plots.
    frequency : int
        Log every N iterations.
    overwrite : bool
        If True, overwrite the same file each time (useful for live monitoring).
        If False, append the iteration index to the filename so every snapshot
        is preserved (e.g., refractive_index_000100.png).
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
    ...     lr = cosine_annealing_lr(iteration, lr_max=10.0, n_steps=n_epochs)
    ...     optic, loss = train_step(optic, ...)
    ...     logger.on_iteration(iteration, optic, {"loss": loss, "lr": lr})
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        self._history: dict[str, list[float]] = {}
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def on_iteration(
        self,
        iteration: int,
        optic: RefractiveOptic,
        metrics: dict[str, float],
    ) -> None:
        """Called each iteration by the training loop.

        Saves a mid-plane refractive index cross-section and a metrics plot
        according to the configured frequency and overwrite settings.

        Parameters
        ----------
        iteration : int
            Current iteration index (0-based).
        optic : RefractiveOptic
            Current state of the optic being optimized.
        metrics : dict[str, float]
            Scalar values to record, e.g. {"loss": 1.23, "lr": 0.05}.
            All keys are plotted together on the metrics chart.
        """
        for key, value in metrics.items():
            self._history.setdefault(key, []).append(value)

        if iteration % self._config.frequency != 0:
            return

        suffix = "" if self._config.overwrite else f"_{iteration:06d}"
        self._save_refractive_index_slice(optic, suffix)
        self._save_metrics_curve(suffix)

    def _filepath(self, name: str, suffix: str, ext: str) -> Path:
        return self._config.output_dir / f"{name}{suffix}.{ext}"

    def _save_refractive_index_slice(self, optic: RefractiveOptic, suffix: str) -> None:
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

    def _save_metrics_curve(self, suffix: str) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("agg")

        for key, values in self._history.items():
            fig, ax = plt.subplots()
            ax.plot(values)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(key)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            fig.savefig(self._filepath(key, suffix, "png"))
            plt.close(fig)
