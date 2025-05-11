import h5py
from pathlib import Path
from puma import Histogram, HistogramPlot
from ftag import Cuts, Flavours, Label
import numpy as np
import fnmatch
from collections import defaultdict
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
        sample_path, 
        output_path,
        plot_kwargs={},
        global_cuts=None,
        all_flavours = ['bjets', 'cjets', 'ujets', 'taujets'],
        num_jets=100_000,
        plot_pattern=None,
        ):
        self.sample_path = Path(sample_path)
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
        
    def plot(self, norm=False, logy=False):
        
        
        if norm and logy:
            ylabel = "Number of events (log)"
            detail_dir = "norm_log"
        elif norm:
            ylabel = "Number of events"
            detail_dir = "norm"
        elif logy:
            ylabel = "Number of events (log)"
            detail_dir = "log"
        else:
            ylabel = "Number of events"
            detail_dir = "raw"

        self.output_path.mkdir(parents=True, exist_ok=True)
        
        h5file = h5py.File(self.sample_path, "r")
        flavours = [Flavours[f] for f in self.all_flavours]
        jets = h5file['jets'][:self.num_jets]
        if self.cuts:
            _, jets = self.cuts(jets)
        jet_masks_by_flavour = {
            flavours[i].name : flavours[i].cuts(jets)[0] for i in range(len(flavours))
        }

        for key in h5file.keys():
            if not self._passes_selection(key):
                continue
            data = h5file[key][:self.num_jets]
            if self.cuts:
                _, data = self.cuts(data)
            # print(jet_masks_by_flavour['bjets'])
            data_by_flavour = {
                flavour.name: data[jet_masks_by_flavour[flavour.name]]
                for flavour in flavours
            }
            if 'valid' in data.dtype.names:
                data_by_flavour = {
                    f : data_by_flavour[f][data_by_flavour[f]['valid']] 
                    for f in data_by_flavour.keys()
                }
            print("Plotting ", key)
            plot_dir = self.output_path / key / detail_dir



            plot_dir.mkdir(parents=True, exist_ok=True)
            
            for v in data.dtype.names:
                if not self._passes_selection(key, v):
                    continue
                try:
                    bins = percentile_bins(data[v], n_bins=50)
                    plot = HistogramPlot(
                        ylabel=ylabel,
                        xlabel=f"{key} {v}",
                        logy=logy,
                        bins=bins,
                        norm=norm,
                        **self.plot_kwargs,
                        
                    )
                    for flavour in flavours:
                        hist = Histogram(
                            data_by_flavour[flavour.name][v],
                            label=flavour.label,
                            colour=flavour.colour,
                        )
                        plot.add(hist)
                    plot.draw()
                    plot.savefig(plot_dir / f"{key}_{v}.png")
                except Exception as e:
                    print("Error plotting ", key, v)
                    print(e)
                    
                    
                

    def create_input_index(self, dropdown_mode=False):
        from collections import defaultdict
        plot_dir = self.output_path
        detail_dirs = ["raw", "norm", "log", "norm_log"]

        html = ['<!DOCTYPE html>',
                '<html lang="en">',
                '<head>',
                '<meta charset="UTF-8">',
                '<title>Input Plot Index</title>',
                '<style>',
                'body { font-family: sans-serif; margin: 20px; }',
                '.contents { margin-bottom: 40px; }',
                '.grid { display: flex; flex-wrap: wrap; margin: -10px; }',
                '.plot { width: 50%; padding: 10px; box-sizing: border-box; }',
                'img { width: 100%; height: auto; display: block; }',
                'select { margin-top: 5px; }',
                '.variant-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }',
                '@media print { .dropdown { display: none; } }',
                '</style>']

        if dropdown_mode:
            html += ['<script>',
                    'function changePlot(select, id) {',
                    '  const img = document.getElementById(id);',
                    '  img.src = select.value;',
                    '}',
                    '</script>']

        html += ['</head>',
                '<body>',
                '<h1>Input Plot Index</h1>',
                '<div class="contents">',
                '<h2>Contents</h2>',
                '<ul>']

        # key -> variable -> detail_dir -> path
        from collections import defaultdict
        key_variable_map = defaultdict(lambda: defaultdict(dict))

        for key_dir in plot_dir.iterdir():
            if not key_dir.is_dir():
                continue
            for detail in detail_dirs:
                detail_dir = key_dir / detail
                if not detail_dir.is_dir():
                    continue
                for img_path in detail_dir.glob("*.png"):
                    variable = img_path.stem
                    rel_path = img_path.relative_to(plot_dir)
                    key_variable_map[key_dir.name][variable][detail] = rel_path

        # Contents
        for key in sorted(key_variable_map):
            html.append(f'<li><a href="#{key}">{key}</a><ul>')
            for var in sorted(key_variable_map[key]):
                html.append(f'<li><a href="#{key}-{var}">{var}</a></li>')
            html.append('</ul></li>')
        html.append('</ul></div>')

        # Sections with 2-wide layout
        for key in sorted(key_variable_map):
            html.append(f'<div class="plot-section"><h2 id="{key}">{key}</h2><div class="grid">')
            for variable in sorted(key_variable_map[key]):
                plot_id = f"{key}-{variable}"
                plots = key_variable_map[key][variable]
                html.append(f'<div class="plot"><h3 id="{plot_id}">{variable}</h3>')

                if dropdown_mode:
                    default_img = plots.get("raw") or list(plots.values())[0]
                    html.append(f'<select class="dropdown" onchange="changePlot(this, \'{plot_id}-img\')">')
                    for detail in detail_dirs:
                        path = plots.get(detail)
                        if path:
                            html.append(f'<option value="{path}">{detail}</option>')
                    html.append('</select>')
                    html.append(f'<img id="{plot_id}-img" src="{default_img}">')

                else:
                    html.append('<div class="variant-grid">')
                    for detail in detail_dirs:
                        path = plots.get(detail)
                        if path:
                            label = detail.replace('_', ' ').title()
                            html.append(f'<div><b>{label}</b><br><img src="{path}"></div>')
                    html.append('</div>')

                html.append('</div>')
            html.append('</div></div>')

        html.append('</body></html>')
        fname = 'input_index'
        if dropdown_mode:
            fname += '_dropdown'
        fname += '.html'
        index_path = plot_dir / fname
        with open(index_path, "w") as f:
            f.write("\n".join(html))