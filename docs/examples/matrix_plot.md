# Matrix plot

The class `MatshowPlot` in `puma.matshow` is designed to plot matrixes based on `matplotlib`'s `matshow`. 


To set up the inputs for the plots, have a look [here](./index.md).

## Basic plot

The class can plot a matrix stored in a `np.ndarray`:

```python
matrix_plotter = MatshowPlot()
mat = np.random.rand(4, 3)
matrix_plotter.draw(mat)
matrix_plotter.savefig("path/to/save_dir/vanilla_mat.png")
```

## Plot customization

Various aspects of the plot appearance can be customized using the class' arguments:
- `x_ticklabels`: Names of the matrix's columns;
- `x_ticks_rotation`: Rotation of the columns' names with respect to the horizontal direction;
- `y_ticklabels`: Names of the matrix's rows;
- `show_entries`: wether to show or not the matrix entries as text over the matrix's pixels (bins);
- `show_percentage`: If `True`, the entries are formatted as percentages (i.e. numbers in [0,1] are multiplied by 100 and the percentage symbol is appended).
- `text_color_threshold`: threshold on the relative luminance of the background color (i.e. the color of the matrix pixel) after which the overlapped text color switches to black, to allow better readability on lighter background colors. By default is set to 0.408, as per [W3C standards](https://www.w3.org/WAI/GL/wiki/Relative_luminance);
- `colormap`: `pyplot.cm` colormap for the plot;
- `cbar_label`: Label of the colorbar;

## Example

Example without any customization:

<img src=https://github.com/umami-hep/puma/raw/examples-material/vanilla_mat.png width=500>

Example with some customization:

<img src=https://github.com/umami-hep/puma/raw/examples-material/mat_custumized.png width=500>

Code to obtain previous examples:
```py
--8<-- "examples/plot_matshow.py"
```