from __future__ import annotations
from typing import Literal
from dataclasses import dataclass
from pathlib import Path
import yaml
from datetime import datetime

import numpy as np

from ftag import Cuts, Flavour, Flavours
from puma.utils import get_good_colours, logger
from puma.hlplots import Results, Tagger

@dataclass
class DatasetConfig:
    pass
@dataclass
class VariablePlotConfig:
    pass

Source = Literal["salt", "pretrained"]
def get_model_output_path(model_dir, plot_epoch=None, ckpt_info=None, sample='ttbar'):

    ckpt_dir = Path(model_dir) / 'ckpts'
    eval_files = list(ckpt_dir.glob(f'*{sample}*.h5'))
    eval_files = [f for f in eval_files if sample in f.name]

    logger.debug(f"Found {len(eval_files)} eval files for {sample} in {ckpt_dir}")

    if len(eval_files) == 0:
        raise ValueError(f"No eval files found for {sample} in {ckpt_dir}")
    elif len(eval_files) == 1:
        return eval_files[0]
    if plot_epoch:
        # TODO - this is dependent on salt outputinng with zfill(5), which is not ideal, maybe regex...
        epoch_str = str(plot_epoch).zfill(5)
        eval_files = [f for f in eval_files if epoch_str in f.name]
        if len(eval_files) == 0:
            raise ValueError(f"No eval files found for {sample} in {ckpt_dir} at epoch {plot_epoch}")
        elif len(eval_files) == 1:
            return eval_files[0]
        logger.debug(f"Found {len(eval_files)} eval files for {sample} in {ckpt_dir} at epoch {plot_epoch}")

    if ckpt_info:
        eval_files = [f for f in eval_files if ckpt_info in f.name]
        if len(eval_files) == 0:
            raise ValueError(f"No eval files found for {sample} in {ckpt_dir} with ckpt_info {ckpt_info}")
        elif len(eval_files) == 1:
            return eval_files[0]
        
        # logger.debug(f"Found {len(eval_files)} eval files for {sample} in {ckpt_dir} with ckpt_info {ckpt_info}")
        raise ValueError(f"Multiple eval files found for {sample} in {ckpt_dir} with ckpt_info {ckpt_info}")

@dataclass 
class ModelConfig:
    name: str
    source: Source
    
    
    model_key: str
    f_c: float
    f_b: float
    style: dict[str, dict[str, str]]

    # For salt models
    save_dir: str = None
    id: str = None
    plot_epoch: int = None
    ckpt_info: str = ""

    sample_paths: dict[str, str] = None

    cuts: Cuts = None
    flavours: list[Flavour] = None
    
    
    tagger : Tagger = None
    eval_path : Path = None
    @property
    def model_path(self):
        return Path(self.save_dir)/self.id

    def __post_init__(self):
        if not self.flavours:
            self.flavours = [Flavours[k] for k in ["bjets", "cjets", "ujets"]]
        else:
            self.flavours = [Flavours[k] for k in self.flavours]
        
        if not self.cuts:
            self.cuts = Cuts.empty()
        else:
            self.cuts = Cuts.from_list(self.cuts)
    def _get_salt_eval_path(self, plt_cfg : PlotConfig):
        # self.eval_path = get_model_output_path(self.model_path, self.plot_epoch, self.ckpt_info, plt_cfg.sample)
        ckpt_dir = Path(self.model_path) / 'ckpts'
        eval_files = list(ckpt_dir.glob(f'*{plt_cfg.sample}*.h5'))
        eval_files = [f for f in eval_files if plt_cfg.sample in f.name]

        logger.debug(f"Found {len(eval_files)} eval files for {plt_cfg.sample} in {ckpt_dir}")

        if len(eval_files) == 0:
            raise ValueError(f"No eval files found for {plt_cfg.sample} in {ckpt_dir}")
        elif len(eval_files) == 1:
            return eval_files[0]
        if self.plot_epoch:
            # TODO - this is dependent on salt outputinng with zfill(5), which is not ideal, maybe regex...
            epoch_str = str(self.plot_epoch).zfill(5)
            eval_files = [f for f in eval_files if epoch_str in f.name]
            if len(eval_files) == 0:
                raise ValueError(f"No eval files found for {plt_cfg.sample} in {ckpt_dir} at epoch {self.plot_epoch}")
            elif len(eval_files) == 1:
                return eval_files[0]
            logger.debug(f"Found {len(eval_files)} eval files for {plt_cfg.sample} in {ckpt_dir} at epoch {self.plot_epoch}")

        if self.ckpt_info:
            eval_files = [f for f in eval_files if self.ckpt_info in f.name]
            if len(eval_files) == 0:
                raise ValueError(f"No eval files found for {plt_cfg.sample} in {ckpt_dir} with ckpt_info {self.ckpt_info}")
            elif len(eval_files) == 1:
                return eval_files[0]
            
            # logger.debug(f"Found {len(eval_files)} eval files for {sample} in {ckpt_dir} with ckpt_info {ckpt_info}")
            raise ValueError(f"Multiple eval files found for {plt_cfg.sample} in {ckpt_dir} with ckpt_info {self.ckpt_info}")

    
        return self.eval_path
    def _load_pretrained_model(self, plot_config : PlotConfig):
        pass
    def load_model(self, plot_config : PlotConfig):
        
        if self.source == "salt":
            self.eval_path =  self._get_salt_eval_path(plot_config)
        elif self.source == "pretrained":
            if not self.sample_paths:
                raise ValueError(f"Must specify sample_paths for pretrained model {self.name}")
            if plot_config.sample not in self.sample_paths:
                raise ValueError(f"Sample {plot_config.sample} not found in sample paths for model {self.name}")
            self.eval_path = self.sample_paths[plot_config.sample]
            # return self._load_pretrained_model(sample)
        else:
            raise ValueError(f"Unknown source {self.source}, must be one of 'salt' or 'pretrained'")
        self.tagger = Tagger(
            name = self.model_key,
            f_c = self.f_c,
            f_b = self.f_b,
            reference=plot_config.denominator==self.name,
            output_nodes =self.flavours,
            **self.style,
        )

    @classmethod
    def get_models(cls, path : Path, models_to_get: list[str] = None):
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        models = {model_key : cls(name=model_key, **model) for model_key, model in config["models"].items()}

        if models_to_get:
            models = {key : model for key, model in models.items() if key in models_to_get}

        model_colours = [model.style.get("colour", "auto") for model in models.values()]
        good_colours = iter(get_good_colours())
        # TODO auto assign good colours when we load...
        for key, model in models.items():
            if model.save_dir and not model.model_path.exists():
                raise FileNotFoundError(f"Path {model.model_path} does not exist")
            # model.
            # if model.style
        return models

@dataclass
class PlotConfig:
    config_path: Path
    plot_dir: Path

    samples: dict[str, dict[str, str]]
    models_config: Path
    models: list[str]
    denominator: str

    roc_plots: dict[str, dict] = None
    fracscan_plots: dict[str, dict] = None

    signal: str = None
    sample: str = None

    num_jets: int = None

    results: Results = None
    default_second_atlas_tag: str = None
    loaded_models: dict[str, ModelConfig] = None
    plot_dir_final: Path = None

    def __post_init__(self):
        # Define a plot directory based on the plot config file name, and a date time
        date_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir_name = self.config_path.stem + "_" + date_time_file
        self.plot_dir_final = Path(self.plot_dir) / plot_dir_name

    @classmethod
    def load_config(cls, path : Path):
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        # If the model config is not an absolute path, assume it is relative to the config file
        if config["models_config"].startswith("/"):
            config["models_config"] = Path(config["models_config"])
        else:
            config["models_config"] = path.parent / config["models_config"]
        
        return cls(config_path=path, **config)
    
    def _get_sample_cut_str(self):
        sample_cuts = Cuts.from_list(self.samples[self.sample].get("cuts", []))
        cut_str = ""

        OP_INV = { '<' : '>', '>' : '<', '<=' : '>=', '>=' : '<='}

        # Currently only for big two cuts... But, perhaps worth doing a general case, as we
        # can never have more than 2 cuts on a variables? 
        if len(pt_cuts := [cut for cut in sample_cuts if cut.variable == 'pt']) == 1:
            cut_str += f"$p_{{T}}$ {pt_cuts[0].operator} {int(pt_cuts[0].value/1000)} GeV   "
        elif len(pt_cuts) == 2:
            min_idx = np.argmin([cut.value for cut in pt_cuts])

            cut_str += f"{int(pt_cuts[min_idx].value/1000)} {OP_INV[pt_cuts[min_idx].operator]} "\
                f"$ p_{{T}}$ {pt_cuts[1-min_idx].operator} {int(pt_cuts[1-min_idx].value/1000)} GeV   "
        elif len(pt_cuts) != 0:
            raise ValueError(f"Invalid number of pt cuts {len(pt_cuts)}")
        if len(eta_cuts := [cut for cut in sample_cuts if cut.variable=='eta']) == 1:
            cut_str += f"$\eta$ {eta_cuts[0].operator} {eta_cuts[0].value}"
        elif len(eta_cuts) == 2:
            min_idx = np.argmin([cut.value for cut in eta_cuts])
            if len(set([abs(cut.value) for cut in eta_cuts])) == 1:
                cut_str += f"$|\eta|$ {eta_cuts[1-min_idx].operator} {eta_cuts[1-min_idx].value}"
            else:
                cut_str += f"{eta_cuts[min_idx].value} {OP_INV[eta_cuts[min_idx].operator]}"\
                    f"$ \eta${eta_cuts[1-min_idx].operator} {eta_cuts[1-min_idx].value}"
        elif len(eta_cuts) != 0:
            raise ValueError(f"Invalid number of eta cuts {len(eta_cuts)}")
        
        for cut in sample_cuts:
            if cut.variable not in ['pt', 'eta']:
                cut_str += str(cut)
        return cut_str
    
    def get_results(self):
        '''Creates the high-level 'Results' object from the config file, using the previously
        set signal and sample. Iterates and loads all models in the config file, and adds them
        '''
        self.loaded_models = ModelConfig.get_models(self.models_config, self.models)
        self.default_second_atlas_tag = self.samples[self.sample]["latex"] + "   " + self._get_sample_cut_str()
        sample_cuts = Cuts.from_list(self.samples[self.sample].get("cuts", []))
    

        
        results = Results(atlas_first_tag="Simulation Internal",
                            atlas_second_tag=self.default_second_atlas_tag,
                          signal=self.signal,
                          sample=self.sample,)
        
        final_plot_dir = self.plot_dir_final / f"{self.signal}_tagging" / self.sample
        final_plot_dir.mkdir(parents=True, exist_ok=True)
        results.output_dir = final_plot_dir
        for key, model in self.loaded_models.items():
            model.load_model(self)
            results.add_taggers_from_file(
                [model.tagger],
                model.eval_path,
                cuts=sample_cuts+model.cuts,
                num_jets=self.num_jets,)
        self.results = results
        

