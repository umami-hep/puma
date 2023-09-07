from __future__ import annotations
from typing import Literal
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import yaml
from datetime import datetime
from ftag.flavour import FlavourContainer
from ftag import flavour

import numpy as np
import h5py 

from ftag import Cuts, Flavour, Flavours
from ftag.hdf5 import H5Reader
from puma.utils import get_good_colours, logger
from puma.hlplots import Results, Tagger

# def get_model_output_path(model_dir, plot_epoch=None, ckpt_info=None, sample='ttbar'):

#     ckpt_dir = Path(model_dir) / 'ckpts'
#     eval_files = list(ckpt_dir.glob(f'*{sample}*.h5'))
#     eval_files = [f for f in eval_files if sample in f.name]

#     logger.debug(f"Found {len(eval_files)} eval files for {sample} in {ckpt_dir}")

#     if len(eval_files) == 0:
#         raise ValueError(f"No eval files found for {sample} in {ckpt_dir}")
#     elif len(eval_files) == 1:
#         return eval_files[0]
#     if plot_epoch:
#         # TODO - this is dependent on salt outputinng with zfill(5), which is not ideal, maybe regex...
#         epoch_str = str(plot_epoch).zfill(5)
#         eval_files = [f for f in eval_files if epoch_str in f.name]
#         if len(eval_files) == 0:
#             raise ValueError(f"No eval files found for {sample} in {ckpt_dir} at epoch {plot_epoch}")
#         elif len(eval_files) == 1:
#             return eval_files[0]
#         logger.debug(f"Found {len(eval_files)} eval files for {sample} in {ckpt_dir} at epoch {plot_epoch}")

#     if ckpt_info:
#         eval_files = [f for f in eval_files if ckpt_info in f.name]
#         if len(eval_files) == 0:
#             raise ValueError(f"No eval files found for {sample} in {ckpt_dir} with ckpt_info {ckpt_info}")
#         elif len(eval_files) == 1:
#             return eval_files[0]
        
#         # logger.debug(f"Found {len(eval_files)} eval files for {sample} in {ckpt_dir} with ckpt_info {ckpt_info}")
#         raise ValueError(f"Multiple eval files found for {sample} in {ckpt_dir} with ckpt_info {ckpt_info}")

def fill_colours(colours, warn=True):
    '''Fills in any colours which are missing with 'good' colours
    colours should be an array like [red, None, green, None, None],
    where all 'None' colours will be filled in with good colours.

    Parameters:
    -----------
    colours : list[str]
        List of colours, with None for any missing colours
    warn : bool, optional
        If true, will warn the user if there is a mix of defined colours, and None, as this
        could result in multiple colours similar to each other
    '''
    if warn and not (all([c == None for c in colours]) or all([c != None for c in colours])):
        logger.warning(all([c == None for c in colours]))
        logger.warning(all([c != None for c in colours]))
        logger.warning("Mix of defined colours and None, this could result in multiple colours similar to each other")
    good_colours = iter(get_good_colours())
    for i, colour in enumerate(colours):
        if colour is None:
            try:
                c = next(good_colours)
                while c in colours:
                    c = next(good_colours)
            except StopIteration:
                raise ValueError(f"Ran out of good colours, need more than {len(colours)} colours")
            colours[i] = c
    return colours
            
@dataclass
class DatasetConfig:
    name: str
    
    path: dict[str, Path]
    style: dict[str, dict[str, str]] = None

    def __post_init__(self):
        self.path = {k : Path(v) for k, v in self.path.items()}
        for path in self.path.values():
            if not path.exists():
                raise FileNotFoundError(f"For dataset {self.name}, sample path {path} does not exist")
        if not self.style:
            self.style = {'label' : self.name}
    
    @classmethod
    def get_datasets(cls, path : Path, datasets_to_get: list[str] = None):
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        # maybe just config.items() ??
        logger.info("Loading datasets: ", config["datasets"])
        datasets = {dataset_key : cls(name=dataset_key, **dataset) for dataset_key, dataset in config["datasets"].items() if dataset_key in datasets_to_get}

        dataset_colours = [model.style.get("colour", None) for model in datasets.values()]
        dataset_colours = fill_colours(dataset_colours)
        for  (key, dataset), new_colour  in zip(datasets.items(), dataset_colours):
            dataset.style["colour"] = new_colour

        return datasets
    
    @property
    def samples(self):
        if isinstance(self.path, dict):
            return list(self.path.keys())
        else:
            return None
    
    def __getitem__(self, sample):
        if not sample in self.samples:
            raise ValueError(f"Sample {sample} not found in dataset {self.name}, available samples are {self.samples}")
        return self.path[sample]
    
    def load_sample(self, sample, cuts=None, keys=None, njets=None, flavours=None):
        '''Loads all of a dataset sample into a dictionary, with keys being the group names.
        '''
        path = self[sample]
        with h5py.File(path, "r") as f:
            logger.info(f.keys())
            logger.info(keys)
            if keys:
                data = {key : f[key] for key in keys}
            else:
                data = {key : f[key] for key in f.keys()}
            if not njets:
                njets = data[list(data.keys())[0]].shape[0]
            # Load everything into numpy array
            data = {key : np.array(data[key][:njets]) for key in data.keys()}
        # apply cuts...
        if cuts:
            logger.info(cuts)
            jets = data['jets']
            idx = cuts(jets).idx
            for key, d in data.items():
                data[key] = d[idx]
        else:
            logger.warning("No cuts applied to dataset")
        if flavours:
            jets = data['jets']

            per_flav_idx = {f.name : f.cuts(jets).idx for f in flavours}
            
            for key, d in data.items():
                per_flav = {f.name : d[per_flav_idx[f.name]] for f in flavours}
                if 'valid' in d.dtype.names:
                    logger.info(f"Selecting only valid {key}")
                    for f in flavours:
                        per_flav[f.name] = per_flav[f.name][per_flav[f.name]['valid']]
                    
                data[key] = per_flav
        else:
            # select only valid
            for key in keys:
                    
                if 'valid' in data[key].dtype.names:
                    logger.info(f"Selecting only valid {key}")
                    data[key] = data[key][data[key]['valid']]
        return data
        

@dataclass
class VariablePlotConfig:
    config_path: Path
    plot_dir: Path
    plot_timestamp: bool
    samples: dict[str, dict[str, str]]
    datasets_config: Path
    denominator: str
    variables: dict[str, list[str]]
    
    plots : dict[str, dict[str, dict]] 

    global_plot_kwargs : dict[str, dict[str, str]] = None

    flavours: list[Flavour] = None
    flavours_file: Path = None
    num_jets: int = None
    keys: list[str] = None

    plot_dir_final: Path = None
    sample: str = None
    datasets: dict[str, DatasetConfig] = None
    loaded_datasets: dict[str, dict[str, np.ndarray]] = None

    def __post_init__(self):
        self.plot_dir = Path(self.plot_dir)
        self.datasets_config = Path(self.datasets_config)
        if not self.flavours:
            self.flavours = ["bjets", "cjets", "ujets"]

        if self.flavours_file:
            with open(self.flavours_file) as f:
                flavours = yaml.safe_load(f)
                # flavours_dict = 
                Flavours = FlavourContainer({f["name"]: Flavour(cuts=Cuts.from_list(f.pop("cuts")), **f) for f in flavours})
        else:
            Flavours = flavour.Flavours
        self.flavours = [Flavours[k] for k in self.flavours]
        logger.info(f"KEYS: {self.keys}")
        if not self.global_plot_kwargs:
            self.global_plot_kwargs = {}
        # if not self.keys:  
        #     self.keys = ["jets"]
        if self.plot_timestamp:
            date_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir_name = self.config_path.stem + "_" + date_time_file
            self.plot_dir_final = Path(self.plot_dir) / plot_dir_name
        else:
            self.plot_dir_final = self.plot_dir / self.config_path.stem

    @classmethod
    def load_config(cls, path : Path):
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        # If the model config is not an absolute path, assume it is relative to the config file
        if config["datasets_config"].startswith("/"):
            config["datasets_config"] = Path(config["datasets_config"])
        else:
            config["datasets_config"] = path.parent / config["datasets_config"]
        
        return cls(config_path=path, **config)

    def load_datasets(self):
        if not self.sample:
            raise ValueError("Must specify sample to load datasets")
        self.datasets = DatasetConfig.get_datasets(self.datasets_config, self.datasets)
        logger.info(f"Sample: {self.samples[self.sample]}" )
        sample_cuts = Cuts.from_list(self.samples[self.sample].get("cuts", []))
        self.loaded_datasets = {key : dataset.load_sample(self.sample, cuts=sample_cuts, keys=set(self.keys+['jets']), njets=self.num_jets, flavours=self.flavours) for key, dataset in self.datasets.items()}
    

Source = Literal["salt", "pretrained"]
@dataclass 
class ModelConfig:
    name: str
    source: Source
    
    f_c: float
    f_b: float
    style: dict[str, dict[str, str]]
    model_key: str = None

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
        ckpt_dir = Path(self.model_path) / 'ckpts'
        eval_files = list(ckpt_dir.glob(f'*{plt_cfg.sample}*.h5'))
        eval_files = [f for f in eval_files if plt_cfg.sample in f.name]

        logger.debug(f"Found {len(eval_files)} eval files for {plt_cfg.sample} in {ckpt_dir}")

        if len(eval_files) == 0:
            raise ValueError(f"No eval files found for {plt_cfg.sample} in {ckpt_dir}")
        elif len(eval_files) == 1:
            return eval_files[0]
        if self.plot_epoch:
            # TODO - this is dependent on salt outputinng with zfill(3), which is not ideal, maybe regex...
            epoch_str = str(self.plot_epoch).zfill(3)
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

    def _get_model_key(self, eval_path : Path):
        if self.model_key:
            return self.model_key
        # TODO actually check this works properly
        reader = H5Reader(eval_path)
        jet_vars = reader.dtypes()['jets'].names
        req_keys = [f'_p{flav.name[:-4]}' for flav in self.flavours]
        # tagger_suffixes = ['_pb', '_pc', '_pu']
        potential_taggers = {}

        # Identify potential taggers
        for var in jet_vars:
            for suffix in req_keys:
                if var.endswith(suffix):
                    base_name = var.rsplit(suffix, 1)[0]
                    if base_name in potential_taggers:
                        potential_taggers[base_name].append(suffix)
                    else:
                        potential_taggers[base_name] = [suffix]

        # Check if any base name has all three suffixes
        valid_taggers = [base for base, suffixes in potential_taggers.items() if set(suffixes) == set(req_keys)]

        if len(valid_taggers) == 0:
            raise ValueError("No valid tagger found.")
        elif len(valid_taggers) > 1:
            raise ValueError(f"Multiple valid taggers found: {', '.join(valid_taggers)}")
        else:
            return valid_taggers[0]
        

    def load_model(self, plot_config : PlotConfig):
        
        if self.source == "salt":
            self.eval_path =  self._get_salt_eval_path(plot_config)
        elif self.source == "pretrained":
            if not self.sample_paths:
                raise ValueError(f"Must specify sample_paths for pretrained model {self.name}")
            if plot_config.sample not in self.sample_paths:
                raise ValueError(f"Sample {plot_config.sample} not found in sample paths for model {self.name}")
            self.eval_path = self.sample_paths[plot_config.sample]
        else:
            raise ValueError(f"Unknown source {self.source}, must be one of 'salt' or 'pretrained'")
        
        
        
        self.tagger = Tagger(
            name = self._get_model_key(self.eval_path),
            f_c = self.f_c,
            f_b = self.f_b,
            reference=plot_config.denominator==self.name,
            output_nodes =self.flavours,
            **self.style,
            yaml_name=self.name,
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

        model_colours = [model.style.get("colour", None) for model in models.values()]
        model_colours = fill_colours(model_colours)
        for  (key, model), new_colour  in zip(models.items(), model_colours):
            model.style["colour"] =new_colour

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

    global_plot_kwargs : dict[str, dict[str, str]] = None


    roc_plots: dict[str, dict] = None
    fracscan_plots: dict[str, dict] = None
    disc_plots: dict[str, dict] = None
    prob_plots: dict[str, dict] = None
    eff_vs_var_plots: dict[str, dict] = None

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
        if not self.global_plot_kwargs:
            self.global_plot_kwargs = {}
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

        # maybe ott, but allows for ' 20 < pT < 250' and '|eta| <2.5' in the same cut string
        # rather than   'pT < 250, pT > 20, eta > -2.5, eta < 2.5, ...'
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
    
    def get_results(self, perf_var='pt'):
        '''Creates the high-level 'Results' object from the config file, using the previously
        set signal and sample. Iterates and loads all models in the config file, and adds them
        '''
        self.loaded_models = ModelConfig.get_models(self.models_config, self.models)
        self.default_second_atlas_tag = self.samples[self.sample]["latex"] + "   " + self._get_sample_cut_str()
        sample_cuts = Cuts.from_list(self.samples[self.sample].get("cuts", []))
    

        
        results = Results(atlas_first_tag="Simulation Internal",
                            atlas_second_tag=self.default_second_atlas_tag,
                          signal=self.signal,
                          sample=self.sample,
                          perf_var=perf_var)
        
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
        

