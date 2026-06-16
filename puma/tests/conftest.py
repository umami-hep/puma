"""Pytest configuration for the puma test suite."""

from __future__ import annotations

from puma.tests._image_utils import set_update_reference_plots


def pytest_addoption(parser):
    """Register the --update-reference-plots command-line flag."""
    parser.addoption(
        "--update-reference-plots",
        action="store_true",
        default=False,
        help=(
            "Overwrite the expected/reference plots with the freshly generated ones "
            "instead of only comparing against them."
        ),
    )


def pytest_configure(config):
    """Propagate the --update-reference-plots flag to the image-comparison helper."""
    set_update_reference_plots(config.getoption("--update-reference-plots"))
