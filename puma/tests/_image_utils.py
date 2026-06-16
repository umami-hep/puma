"""Shared helpers for image-comparison tests.

Plotting tests render a figure and compare it pixel-wise against a stored reference
image under ``puma/tests/**/expected_plots/``. Pass ``--update-reference-plots`` to
pytest to regenerate the reference images instead of only comparing against them
(see ``puma/tests/conftest.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from matplotlib.testing.compare import compare_images

# Toggled on by conftest.py when --update-reference-plots is passed to pytest.
_UPDATE_REFERENCE_PLOTS = False


def set_update_reference_plots(value: bool) -> None:
    """Enable/disable overwriting reference plots with freshly generated ones."""
    global _UPDATE_REFERENCE_PLOTS  # noqa: PLW0603
    _UPDATE_REFERENCE_PLOTS = bool(value)


def assert_plot_matches(
    actual_dir: str | Path,
    expected_dir: str | Path,
    name: str,
    tol: float = 1,
) -> None:
    """Compare a generated plot against its reference image.

    Parameters
    ----------
    actual_dir : str | Path
        Directory containing the freshly generated plot.
    expected_dir : str | Path
        Directory containing the reference (expected) plot.
    name : str
        File name of the plot, shared between both directories.
    tol : float, optional
        Pixel tolerance passed to ``matplotlib.testing.compare.compare_images``,
        by default 1.

    Notes
    -----
    When ``--update-reference-plots`` is active, the generated plot is copied over
    the reference first, so the comparison then trivially passes and the reference
    image is updated in place.
    """
    actual = Path(actual_dir) / name
    expected = Path(expected_dir) / name
    if _UPDATE_REFERENCE_PLOTS:
        shutil.copyfile(actual, expected)
    assert compare_images(str(actual), str(expected), tol=tol) is None
