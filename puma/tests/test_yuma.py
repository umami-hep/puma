#!/usr/bin/env python
"""Unit test script for the functions in hlplots/tagger.py."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import yaml
from ftag import Flavours, get_mock_file
from ftag.hdf5 import structured_from_dict

from puma.hlplots import PlotConfig, get_signals
from puma.hlplots.plot_ftag import main
from puma.hlplots.yutils import get_tagger_name


class TestYutils(unittest.TestCase):
    def setUp(self):
        self.flavours = [Flavours[f] for f in ["ujets", "cjets", "bjets"]]

    def testGetIncludeTaggers(self):
        plt_cfg = Path(__file__).parent.parent.parent / "examples/plt_cfg.yaml"
        taggers = Path(__file__).parent.parent.parent / "examples/taggers.yaml"
        with open(plt_cfg) as f:
            plt_cfg = yaml.safe_load(f)
        with open(taggers) as f:
            taggers = yaml.safe_load(f)

        with tempfile.TemporaryDirectory() as tmp_file:
            fpath1, file = get_mock_file(fname=(Path(tmp_file) / "file1.h5").as_posix())
            taggers["tagger_defaults"]["sample_path"] = fpath1
            taggers["taggers"]["dummy3"]["sample_path"] = fpath1
            updated_taggers = Path(tmp_file) / "taggers.yaml"
            updated_plt_cfg = Path(tmp_file) / "plt_cfg.yaml"

            plt_cfg["roc_plots"][0]["reference"] = "dummyNot"
            plt_cfg["roc_plots"][0]["include_taggers"] = ["dummy1"]

            plt_cfg["taggers_config"] = updated_taggers.as_posix()
            plt_cfg["plot_dir"] = tmp_file + "/plots"

            # write the yaml files
            with open(updated_taggers, "w") as f:
                yaml.dump(taggers, f)
            with open(updated_plt_cfg, "w") as f:
                yaml.dump(plt_cfg, f)

            args = ["--config", updated_plt_cfg.as_posix(), "--signals", "bjets"]
            main(args)

    def testGetTaggerName(self):
        fpath, file = get_mock_file()
        name = get_tagger_name(None, fpath, flavours=self.flavours)
        assert name == "MockTagger"

    def testBreakGetTaggerName(self):
        fpath, file = get_mock_file()
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
                get_tagger_name(None, fname, flavours=self.flavours)

    def testGetSignals(self):
        plt_cfg = Path(__file__).parent.parent.parent / "examples/plt_cfg.yaml"
        plt_cfg = PlotConfig.load_config(plt_cfg)
        valid = get_signals(plt_cfg)
        assert sorted(valid) == ["bjets", "cjets"]


class TestYumaPlots(unittest.TestCase):
    def testArgs(self):
        args = ["--config", "config.yaml", "--signals", "bjets", "--plots", "not_valid"]
        with self.assertRaises(ValueError):
            main(args)

    def testAllPlots(self):
        plt_cfg = Path(__file__).parent.parent.parent / "examples/plt_cfg.yaml"
        with open(plt_cfg) as f:
            plt_cfg = yaml.safe_load(f)

        taggers = Path(__file__).parent.parent.parent / "examples/taggers.yaml"
        with open(taggers) as f:
            taggers = yaml.safe_load(f)

        with tempfile.TemporaryDirectory() as tmp_file:
            fpath1, file = get_mock_file(fname=(Path(tmp_file) / "file1.h5").as_posix())
            fpath2, file = get_mock_file(fname=(Path(tmp_file) / "file2.h5").as_posix())

            taggers["tagger_defaults"]["sample_path"] = fpath1
            taggers["taggers"]["dummy3"]["sample_path"] = fpath2
            updated_taggers = Path(tmp_file) / "taggers.yaml"
            updated_plt_cfg = Path(tmp_file) / "plt_cfg.yaml"
            plt_cfg["taggers_config"] = updated_taggers.as_posix()
            plt_cfg["plot_dir"] = tmp_file + "/plots"

            # write the yaml files
            with open(updated_taggers, "w") as f:
                yaml.dump(taggers, f)
            with open(updated_plt_cfg, "w") as f:
                yaml.dump(plt_cfg, f)

            args = ["--config", updated_plt_cfg.as_posix(), "--signals", "bjets"]
            main(args)

            # Simple check on number of output plots
            out_dir = Path(tmp_file) / "plots" / "plt_cfg"
            btagging = out_dir / "bjets_tagging"
            ctagging = out_dir / "cjets_tagging"
            assert btagging.exists(), "No b-tagging plots produced"
            assert not ctagging.exists(), "No c-tagging plots should have been produced"
            btag_plots = [p.name for p in btagging.glob("*")]
            assert (
                len(btag_plots) == 22
            ), f"Expected 22 b-tagging plot, found {len(btag_plots)}"

            args = [
                "--config",
                updated_plt_cfg.as_posix(),
                "--plots",
                "roc",
                "--signals",
                "cjets",
            ]
            main(args)

            ctag_plots = [p.name for p in ctagging.glob("*")]
            assert ctagging.exists(), "No c-tagging plots produced"
            assert (
                len(ctag_plots) == 1
            ), f"Only expected one c-tagging plot, found {len(ctag_plots)}"

            args = ["--config", updated_plt_cfg.as_posix()]
            main(args)

    def testNoPlots(self):
        plt_cfg = Path(__file__).parent.parent.parent / "examples/plt_cfg.yaml"
        with open(plt_cfg) as f:
            plt_cfg = yaml.safe_load(f)

        taggers = Path(__file__).parent.parent.parent / "examples/taggers.yaml"
        with open(taggers) as f:
            taggers = yaml.safe_load(f)

        with tempfile.TemporaryDirectory() as tmp_file:
            fpath1, file = get_mock_file(fname=(Path(tmp_file) / "file1.h5").as_posix())
            fpath2, file = get_mock_file(fname=(Path(tmp_file) / "file2.h5").as_posix())

            taggers["tagger_defaults"]["sample_path"] = fpath1
            taggers["taggers"]["dummy3"]["sample_path"] = fpath2
            updated_taggers = Path(tmp_file) / "taggers.yaml"
            updated_plt_cfg = Path(tmp_file) / "plt_cfg.yaml"
            plt_cfg["taggers_config"] = updated_taggers.as_posix()
            plt_cfg["plot_dir"] = tmp_file + "/plots"
            for plot_type in [
                "fracscan_plots",
                "disc_plots",
                "prob_plots",
                "eff_vs_var_plots",
            ]:
                plt_cfg[plot_type] = []
            # write the yaml files
            with open(updated_taggers, "w") as f:
                yaml.dump(taggers, f)
            with open(updated_plt_cfg, "w") as f:
                yaml.dump(plt_cfg, f)

            args = ["--config", updated_plt_cfg.as_posix(), "--signals", "bjets"]
            main(args)

            # Simple check on number of output plots
            out_dir = Path(tmp_file) / "plots" / "plt_cfg"
            btagging = out_dir / "bjets_tagging"
            ctagging = out_dir / "cjets_tagging"
            assert btagging.exists(), "No b-tagging plots produced"
            assert not ctagging.exists(), "No c-tagging plots should have been produced"
            btag_plots = [p.name for p in btagging.glob("*")]
            assert (
                len(btag_plots) == 3
            ), f"Expected 3 b-tagging plot, found {len(btag_plots)}"
