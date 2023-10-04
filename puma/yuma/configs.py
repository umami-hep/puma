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
        cidx = 0
        good_colours = list(get_good_colours())
        for d, dset in datasets.items():
            if 'colour' not in dset.style:
                dset.style['colour'] = good_colours[cidx]
                cidx += 1


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
    
def _get_tagger_name(name: str, sample_path : Path, flavours : list[Flavour]):
    if name:
        return name
    # TODO actually check this works properly
    reader = H5Reader(sample_path)
    jet_vars = reader.dtypes()['jets'].names
    req_keys = [f'_p{flav.name[:-4]}' for flav in flavours]
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
@dataclass
class PlotConfig:
    config_path: Path
    plot_dir: Path

    taggers_config: Path 
    taggers: list[str] | list[Tagger]
    reference_tagger: str

    sample: dict[str, str] 

    # global_plot_kwargs : dict[str, dict[str, str]] = None
    results_default : dict[str, dict[str, str]] = None
    timestamp: bool = True

    roc_plots: dict[str, dict] = None
    fracscan_plots: dict[str, dict] = None
    disc_plots: dict[str, dict] = None
    prob_plots: dict[str, dict] = None
    eff_vs_var_plots: dict[str, dict] = None

    signal: str = None

    num_jets: int = None

    results: Results = None
    default_second_atlas_tag: str = None
    plot_dir_final: Path = None

    def __post_init__(self):
        # Define a plot directory based on the plot config file name, and a date time
        plot_dir_name = self.config_path.stem
        if self.timestamp:
            date_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir_name += "_" + date_time_file
        self.plot_dir_final = Path(self.plot_dir) / plot_dir_name
        if not self.results_default:
            self.results_default = {}
        
        with open(self.taggers_config) as f:
            self.taggers_config = yaml.safe_load(f)
        tagger_defaults = self.taggers_config.get("tagger_defaults", {})
        # print(self.taggers_config.get("taggers", {}))
        taggers = self.taggers_config.get("taggers", {})
        assert (self.reference_tagger in taggers), (
                f"Reference tagger {self.reference_tagger} not in taggers config"
        )
        self.taggers = {k : {
            **tagger_defaults, 
            **t, 
            "reference" : k==self.reference_tagger,
            "yaml_name" : k,
        } for k, t in taggers.items() if k in self.taggers}



        self.roc_plots = self.roc_plots or {}
        self.fracscan_plots = self.fracscan_plots or {}
        self.disc_plots = self.disc_plots or {}
        self.prob_plots = self.prob_plots or {}
        self.eff_vs_var_plots = self.eff_vs_var_plots or {}


    @classmethod
    def load_config(cls, path : Path):
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        # If the model config is not an absolute path, assume it is relative to the config file
        if config["taggers_config"].startswith("/"):
            config["taggers_config"] = Path(config["taggers_config"])
        else:
            config["taggers_config"] = path.parent / config["taggers_config"]
        
        return cls(config_path=path, **config)
    
    def get_results(self, perf_var='pt'):
        '''Creates the high-level 'Results' object from the config file, using the previously
        set signal and sample. Iterates and loads all models in the config file, and adds them
        '''
        results_default = {
            'atlas_first_tag' : "Simulation Internal",
            'atlas_second_tag' : "$\sqrt{s} = 13.0 $ TeV",
            'global_cuts' : Cuts.empty(),
        }
        results_default.update(self.results_default)
        
        results_default['atlas_second_tag']  += '\n'+self.sample.get("str", "")
        # Store default tag incase other plots need to temporarily modify it
        self.default_second_atlas_tag = results_default['atlas_second_tag']

        sample_cuts = Cuts.from_list(self.sample.get("cuts", []))
        results_default['global_cuts'] = results_default['global_cuts'] + sample_cuts

        results = Results(atlas_first_tag="Simulation Internal",
                            atlas_second_tag=self.default_second_atlas_tag,
                          signal=self.signal,
                          sample=self.sample['name'],
                          perf_var=perf_var,
                          global_cuts=sample_cuts,
                          num_jets=self.num_jets)
        
        good_colours = get_good_colours()
        col_idx = 0
        # Add taggers to results, then bulk load
        for t in self.taggers.values():
            # Allows automatic selection of tagger name in eval files
            t['name'] = _get_tagger_name(
                t.get("name", None), 
                t['sample_path'], 
                results.flavours)
            # Enforces a tagger to have same colour across multiple plots
            if 'colour' not in t:
                t['colour'] = good_colours[col_idx]
                col_idx += 1
            results.add(Tagger(**t))

        results.load()
        
        final_plot_dir = self.plot_dir_final / f"{self.signal}_tagging"
        final_plot_dir.mkdir(parents=True, exist_ok=True)
        
        results.output_dir = final_plot_dir
        self.results = results
        

