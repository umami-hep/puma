"""Unit test script for the functions in hlplots/tagger.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import yaml
from ftag import Flavours, get_mock_file
from ftag.hdf5 import structured_from_dict
from yamlinclude import YamlIncludeConstructor

from puma.hlplots.yuma import YumaConfig, main
from puma.hlplots.yutils import get_tagger_name

EXAMPLES = Path(__file__).parents[3] / "examples"


def load_no_include(plt_cfg, taggers):
    def dummy_inc(loader, node):  # noqa: ARG001
        return node.value

    # Don't load the taggers
    yaml.SafeLoader.add_constructor("!include", dummy_inc)

    plt_cfg = EXAMPLES / "plt_cfg.yaml"
    with open(plt_cfg) as f:
        plt_cfg = yaml.safe_load(f)
    taggers = EXAMPLES / "taggers.yaml"
    with open(taggers) as f:
        taggers = yaml.safe_load(f)

    return plt_cfg, taggers


class TestYutils(unittest.TestCase):
    def setUp(self):
        self.flavours = [Flavours[f] for f in ["ujets", "cjets", "bjets"]]
        # support inclusion of yaml files in the config dir
        YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir=EXAMPLES)

    def testGetIncludeTaggers(self):
        plt_cfg = EXAMPLES / "plt_cfg.yaml"
        taggers = EXAMPLES / "taggers.yaml"
        plt_cfg, taggers = load_no_include(plt_cfg, taggers)

        with tempfile.TemporaryDirectory() as tmp_file:
            fpath1, _file = get_mock_file(fname=(Path(tmp_file) / "file1.h5").as_posix())
            taggers["dummy1"]["sample_path"] = fpath1
            taggers["dummy2"]["sample_path"] = fpath1
            taggers["dummy3"]["sample_path"] = fpath1
            updated_plt_cfg = Path(tmp_file) / "plt_cfg.yaml"

            plt_cfg["plots"]["roc"][0]["reference"] = "dummyNot"
            plt_cfg["plots"]["roc"][0]["include_taggers"] = ["dummy1"]

            plt_cfg["plot_dir"] = tmp_file + "/plots"
            plt_cfg["taggers_config"] = taggers

            with open(updated_plt_cfg, "w") as f:
                yaml.dump(plt_cfg, f)
            print(plt_cfg)
            args = ["--config", updated_plt_cfg.as_posix(), "--signals", "bjets"]
            main(args)

    def testGetTaggerName(self):
        fpath, _file = get_mock_file()
        name = get_tagger_name(None, fpath, key="TestName1", flavours=self.flavours)
        assert name == "MockTagger"

    def testBreakGetTaggerName(self):
        _fpath, file = get_mock_file()
        updated = {k: file["jets"][k] for k in file["jets"].dtype.names}
        updated["Tagger2_pu"] = updated["MockTagger_pu"]
        updated["Tagger2_pb"] = updated["MockTagger_pb"]
        updated["Tagger2_pc"] = updated["MockTagger_pc"]
        array = structured_from_dict(updated)
        with tempfile.TemporaryDirectory() as tmp_file:
            fname = Path(tmp_file) / "test.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("jets", data=array)
            with self.assertRaises(ValueError):
                get_tagger_name(None, fname, key=None, flavours=self.flavours)

    def testGetSignals(self):
        plt_cfg = EXAMPLES / "plt_cfg.yaml"
        plt_cfg = YumaConfig.load_config(plt_cfg)
        assert sorted(plt_cfg.signals) == ["bjets", "cjets"]


class TestYumaPlots(unittest.TestCase):
    def testArgs(self):
        config_path = str(EXAMPLES / "plt_cfg.yaml")
        args = ["--config", config_path, "--signals", "bjets", "--plots", "not_valid"]
        with self.assertRaises(SystemExit):
            main(args)

    def testCheckConfig(self):
        with tempfile.TemporaryDirectory() as tmp_file:
            cpath = Path(tmp_file) / "dummyconfig.yaml"
            tmp_config = {
                "plot_dir": "test/dir",
                "taggers_config": {"tagger1": {"sample_path": "dummy1"}},
                "results_config": {"sample": "ttbar"},
                "plots": {
                    "roc": [
                        {
                            "signal": "bjets",
                            "include_taggers": ["tagger1"],
                            "efficiency": 0.8,  # Shouldn't be here
                        }
                    ]
                },
            }
            with open(cpath, "w") as f:
                yaml.dump(tmp_config, f)
            args = ["--config", cpath.as_posix(), "--signals", "bjets"]
            with self.assertRaises(ValueError):
                main(args)

    def testAllPlots(self):
        plt_cfg = EXAMPLES / "plt_cfg.yaml"
        taggers = EXAMPLES / "taggers.yaml"
        plt_cfg, taggers = load_no_include(plt_cfg, taggers)

        with tempfile.TemporaryDirectory() as tmp_file:
            fpath1, _file = get_mock_file(fname=(Path(tmp_file) / "file1.h5").as_posix())
            fpath2, _file = get_mock_file(fname=(Path(tmp_file) / "file2.h5").as_posix())
            taggers["dummy1"]["sample_path"] = fpath1
            taggers["dummy2"]["sample_path"] = fpath1
            taggers["dummy3"]["sample_path"] = fpath2
            updated_plt_cfg = Path(tmp_file) / "plt_cfg.yaml"
            plt_cfg["plot_dir"] = tmp_file + "/plots"
            plt_cfg["taggers_config"] = taggers

            with open(updated_plt_cfg, "w") as f:
                yaml.dump(plt_cfg, f)

            args = ["--config", updated_plt_cfg.as_posix(), "--signals", "bjets"]
            main(args)

            # Simple check on number of output plots
            out_dir = Path(tmp_file) / "plots" / "plt_cfg"
            btagging = out_dir / "btag"
            ctagging = out_dir / "ctag"
            assert btagging.exists(), "No b-tagging plots produced"
            assert not ctagging.exists(), "No c-tagging plots should have been produced"
            btag_plots = [p.name for p in btagging.rglob("*.png")]
            assert len(btag_plots) == 22, f"Expected 22 b-tagging plot, found {len(btag_plots)}"

            args = [
                "--config",
                updated_plt_cfg.as_posix(),
                "--plots",
                "roc",
                "--signals",
                "cjets",
            ]
            main(args)

            ctag_plots = [p.name for p in ctagging.rglob("*.png")]
            assert ctagging.exists(), "No c-tagging plots produced"
            assert (
                len(ctag_plots) == 1
            ), f"Only expected one c-tagging plot, found {len(ctag_plots)}: , {ctag_plots}"

            args = ["--config", updated_plt_cfg.as_posix()]
            main(args)

    def testNoPlots(self):
        plt_cfg = EXAMPLES / "plt_cfg.yaml"
        taggers = EXAMPLES / "taggers.yaml"

        plt_cfg, taggers = load_no_include(plt_cfg, taggers)

        with tempfile.TemporaryDirectory() as tmp_file:
            fpath1, _file = get_mock_file(fname=(Path(tmp_file) / "file1.h5").as_posix())
            fpath2, _file = get_mock_file(fname=(Path(tmp_file) / "file2.h5").as_posix())

            taggers["dummy1"]["sample_path"] = fpath1
            taggers["dummy2"]["sample_path"] = fpath1
            taggers["dummy3"]["sample_path"] = fpath2

            updated_plt_cfg = Path(tmp_file) / "plt_cfg.yaml"
            plt_cfg["taggers_config"] = taggers
            plt_cfg["plot_dir"] = tmp_file + "/plots"

            plt_cfg["plots"] = {"roc": plt_cfg["plots"]["roc"]}
            with open(updated_plt_cfg, "w") as f:
                yaml.dump(plt_cfg, f)

            args = ["--config", updated_plt_cfg.as_posix(), "--signals", "bjets"]
            main(args)

            # Simple check on number of output plots
            out_dir = Path(tmp_file) / "plots" / "plt_cfg"
            btagging = out_dir / "btag"
            ctagging = out_dir / "ctag"
            assert btagging.exists(), "No b-tagging plots produced"
            assert not ctagging.exists(), "No c-tagging plots should have been produced"
            btag_plots = [p.name for p in btagging.rglob("*.png")]
            print(btag_plots)
            assert len(btag_plots) == 3, f"Expected 3 b-tagging plot, found {len(btag_plots)}"
