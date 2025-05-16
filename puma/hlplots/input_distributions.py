import h5py
from pathlib import Path
from puma import Histogram, HistogramPlot
from puma.utils import get_good_linestyles
from ftag import Cuts, Flavours, Label
import numpy as np
import fnmatch
from collections import defaultdict
import argparse
import yaml

def percentile_bins(data, lower=0.01, upper=0.99, n_bins=50):
    """
    Create bins based on percentiles of the data.
    
    Parameters:
    - data: The data to create bins for.
    - lower: The lower percentile (default is 0.01).
    - upper: The upper percentile (default is 0.99).
    - n_bins: The number of bins (default is 50).
    
    Returns:
    - bins: The bin edges.
    """
    mask = np.isnan(data) | np.isinf(data)
    lower_bound = np.percentile(data[~mask], lower * 100)
    upper_bound = np.percentile(data[~mask], upper * 100)
    if lower_bound == upper_bound:
        lower_bound = np.min(data[~mask])
        upper_bound = np.max(data[~mask])
    bins = np.linspace(lower_bound, upper_bound, n_bins + 1)
    return bins


class InputDistribution:

    def __init__(
        self, 
        sample_paths : dict[str, str], 
        output_path,
        plot_kwargs={},
        global_cuts=None,
        all_flavours = ['bjets', 'cjets', 'ujets', 'taujets'],
        num_jets=100_000,
        plot_pattern=None,
        ):
        self.sample_paths = {k : Path(v) for k, v in sample_paths.items()}
        self.output_path = Path(output_path)
        self.num_jets = num_jets
        self.cuts = Cuts.empty() if global_cuts is None else Cuts.from_list(global_cuts)
        self.all_flavours = all_flavours

        default_plot_kwargs = {
            'atlas_first_tag': "Simulation Internal",
            'figsize' : (6, 5)
        }
        if 'atlas_third_tag' in plot_kwargs:
            if 'atlas_second_tag' in plot_kwargs:
                plot_kwargs['atlas_second_tag'] += "\n" + plot_kwargs['atlas_third_tag']
            else:
                plot_kwargs['atlas_second_tag'] = plot_kwargs['atlas_third_tag']
            del plot_kwargs['atlas_third_tag']
        default_plot_kwargs.update(plot_kwargs)
        self.plot_kwargs = default_plot_kwargs

        

        if plot_pattern:
            self.plot_pattern = []
            for pattern in plot_pattern:
                
                split = pattern.split('.')
                assert len(split) == 2, "Pattern must be of the form 'key.variable'"
                key, variable = split
                self.plot_pattern.append((key, variable))
        else:
            self.plot_pattern = [('*', '*')]

    

    def _passes_selection(self, group, variable=None):
        for key_pattern, var_pattern in self.plot_pattern:
            if variable is not None:
                if fnmatch.fnmatch(group, key_pattern) and fnmatch.fnmatch(variable, var_pattern):
                    return True
            elif fnmatch.fnmatch(group, key_pattern):
                return True

        return False
    
    def _check_files(self):

        if len(self.sample_paths) == 0:
            raise ValueError("No sample paths provided.")
        if len(self.sample_paths)  == 0:
            return True
        f0 = self.sample_paths[list(self.sample_paths.keys())[0]]
        h50 = h5py.File(f0, "r")
        for _, f in self.sample_paths.items():
            if not f.exists():
                raise ValueError(f"File {f} does not exist.")
            h5 = h5py.File(f, "r")
            if len(h5.keys()) != len(h50.keys()):
                raise ValueError(f"Files {f} and {f0} have different number of groups.")
            for key in h5.keys():
                if key not in h50.keys():
                    raise ValueError(f"Group {key} not found in file {f0}.")
                if len(h5[key].dtype.names) != len(h50[key].dtype.names):
                    raise ValueError(f"Group {key} in file {f} has different number of variables than group in file {f0}.")
        return True

    def plot(self, norm=False, logy=False):
        
        
        if norm and logy:
            ylabel = "Normalized Number of events (log)"
            detail_dir = "norm_log"
        elif norm:
            ylabel = "Normalized Number of events"
            detail_dir = "norm"
        elif logy:
            ylabel = "Number of events (log)"
            detail_dir = "log"
        else:
            ylabel = "Number of events"
            detail_dir = "raw"
        # self._check_files()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        h5files = {k : h5py.File(v, "r") for k, v in self.sample_paths.items()}
        jet_masks_by_flavour_by_file = {}
        h50 = None
        flavours = [Flavours[f] for f in self.all_flavours]
        
        for i, (k, v) in enumerate(h5files.items()):
            if i == 0:
                h50 = v

            jets = v['jets'][:self.num_jets]
                    
            if self.cuts:
                _, jets = self.cuts(jets)
            jet_masks_by_flavour = {
                    flavours[i].name : flavours[i].cuts(jets)[0] for i in range(len(flavours))
            }
            jet_masks_by_flavour_by_file[k] = jet_masks_by_flavour

        linestyles = get_good_linestyles()[:len(h5files)]
        for key in h50.keys():
            if not self._passes_selection(key):
                continue

            data_by_file = {}
            combined_data = []
            for k, h5 in h5files.items():
                print('lol', k)
                data = h5[key][:self.num_jets]
                combined_data.append(data)
                if self.cuts:
                    _, data = self.cuts(data)
            
                data_by_flavour = {
                    flavour.name: data[jet_masks_by_flavour_by_file[k][flavour.name]]
                    for flavour in flavours
                }
                if 'valid' in data.dtype.names:
                    data_by_flavour = {
                        f : data_by_flavour[f][data_by_flavour[f]['valid']] 
                        for f in data_by_flavour.keys()
                    }
                data_by_file[k] = data_by_flavour
            # print(data_by_file['base'].shape)
            combined_data = np.concatenate(combined_data)
            if self.cuts:   
                _, combined_data =  self.cuts(combined_data)
            print("Plotting ", key)
            plot_dir = self.output_path / key / detail_dir



            plot_dir.mkdir(parents=True, exist_ok=True)
            
            for v in combined_data.dtype.names:
                if not self._passes_selection(key, v):
                    continue
                try:
                    bins = percentile_bins(combined_data[v], n_bins=50)
                    plot = HistogramPlot(
                        ylabel=ylabel,
                        xlabel=f"{key} {v}",
                        logy=logy,
                        bins=bins,
                        norm=norm,
                        **self.plot_kwargs,
                        
                    )
                    for i, (k, data_by_flavour) in enumerate(data_by_file.items()):
                        for flavour in flavours:
                            hist = Histogram(
                                data_by_flavour[flavour.name][v],
                                label=flavour.label if i ==0 else None,
                                colour=flavour.colour,
                                linestyle=linestyles[i],
                            )
                            plot.add(hist)
                    if len(data_by_file) > 1:
                        plot.make_linestyle_legend(linestyles, data_by_file.keys())
                    plot.draw()
                    plot.savefig(plot_dir / f"{key}_{v}.png")
                except Exception as e:
                    print("Error plotting ", key, v)
                    print(e)
                    
                    
                



def get_args():
    parser = argparse.ArgumentParser(description="Plot input distributions from HDF5 files.")
    parser.add_argument("--config", '-c', type=str, required=True, help="Paths input distribution config files.")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    id = InputDistribution(**config)

    for [norm, logy] in [(False, False), (True, False), (False, True), (True, True)]:
        id.plot(norm=norm, logy=logy)
