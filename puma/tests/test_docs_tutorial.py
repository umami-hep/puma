from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import h5py
import matplotlib as mpl
import numpy as np
import pytest
from numpy.lib.recfunctions import append_fields, rename_fields

from puma.utils import get_dummy_2_taggers

PYTHON_FENCE_RE = re.compile(
    r"(?P<indent>[ \t]*)```(?P<lang>py|python)\n(?P<code>.*?)(?P=indent)```",
    re.DOTALL,
)
SKIP_COMMENT_RE = re.compile(r"<!--\s*mdtest:\s*skip(?:\s*-\s*.*?)?-->")
TTBAR_EOS_PATH = '"/eos/user/u/umamibot/tutorials/ttbar.h5"'
FLIPPED_PLACEHOLDER_PATH = '"/path/to/tutorial/data/ttbar.h5"'


class PythonBlock(NamedTuple):
    line_number: int
    code: str
    skip: bool


def iter_python_blocks(markdown_path: Path) -> Iterator[PythonBlock]:
    text = markdown_path.read_text(encoding="utf-8")
    for match in PYTHON_FENCE_RE.finditer(text):
        code_start = match.start("code")
        line_number = text.count("\n", 0, code_start) + 1
        preceding_text = text[: match.start()].rstrip()
        previous_line = preceding_text.rsplit("\n", maxsplit=1)[-1]
        yield PythonBlock(
            line_number=line_number,
            code=textwrap.dedent(match.group("code")),
            skip=bool(SKIP_COMMENT_RE.fullmatch(previous_line.strip())),
        )


def build_tutorial_file(path: Path, *, seed: int = 42) -> None:
    jets = get_dummy_2_taggers(size=9_999, add_pt=True, seed=seed)
    jets = rename_fields(
        jets,
        {
            "dips_pu": "dipsLoose20220314v2_pu",
            "dips_pc": "dipsLoose20220314v2_pc",
            "dips_pb": "dipsLoose20220314v2_pb",
        },
    )
    rng = np.random.default_rng(seed)
    jets = append_fields(
        jets,
        [
            "rnnipflip_pu",
            "rnnipflip_pc",
            "rnnipflip_pb",
            "averageInteractionsPerCrossing",
        ],
        [
            np.clip(jets["rnnip_pu"] + rng.normal(0, 0.02, len(jets)), 1e-6, 1),
            np.clip(jets["rnnip_pc"] + rng.normal(0, 0.02, len(jets)), 1e-6, 1),
            np.clip(jets["rnnip_pb"] + rng.normal(0, 0.02, len(jets)), 1e-6, 1),
            rng.uniform(15, 65, len(jets)),
        ],
        usemask=False,
    )
    tracks = np.rec.fromarrays(
        [rng.normal(0, 1, size=(len(jets), 8)).astype("f4")],
        dtype=np.dtype([("IP3D_signed_d0_significance", "f4")]),
    )
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("jets", data=jets)
        h5file.create_dataset("tracks_loose", data=tracks)


def prepare_tutorial_data(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "ttbar": tmp_path / "ttbar.h5",
        "zpext": tmp_path / "zpext.h5",
        "zpext_run3": tmp_path / "zpext_run3.h5",
    }
    build_tutorial_file(paths["ttbar"], seed=42)
    build_tutorial_file(paths["zpext"], seed=43)
    build_tutorial_file(paths["zpext_run3"], seed=44)
    return paths


def rewrite_tutorial_paths(code: str, data_paths: dict[str, Path]) -> str:
    return (
        code.replace(TTBAR_EOS_PATH, repr(data_paths["ttbar"].as_posix()))
        .replace(FLIPPED_PLACEHOLDER_PATH, repr(data_paths["ttbar"].as_posix()))
        .replace('"zpext.h5"', repr(data_paths["zpext"].as_posix()))
        .replace('"zpext_run3.h5"', repr(data_paths["zpext_run3"].as_posix()))
    )


def test_tutorial_python_blocks_execute(tmp_path, monkeypatch):
    """Run executable Python blocks from the plotting tutorial in document order."""
    mpl.use("Agg")
    monkeypatch.chdir(tmp_path)

    tutorial = Path(__file__).parents[2] / "docs/examples/tutorial-plotting.md"
    data_paths = prepare_tutorial_data(tmp_path)
    namespace = {"__name__": "__puma_tutorial_mdtest__"}

    for block in iter_python_blocks(tutorial):
        filename = f"{tutorial}:{block.line_number}"
        code = rewrite_tutorial_paths(block.code, data_paths)
        compiled = compile(code, filename, "exec")
        if block.skip:
            continue
        try:
            exec(compiled, namespace)  # noqa: S102
        except Exception as error:  # noqa: BLE001
            pytest.fail(f"Python block starting at {filename} failed: {error}")
