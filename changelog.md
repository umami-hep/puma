# Changelog

### [Latest]
- Decreased default `figsize` for plots with zero or one ratio panels [#90](https://github.com/umami-hep/puma/pull/90)

### [v0.1.4]
- Renamed the `puma.FractionScan` and `puma.FractionScanPlot` classes to more general `puma.Line2DPlot` and `pumal.Line2D` [#84](https://github.com/umami-hep/puma/pull/84)
- Splitting `force` argument of `set_log()` method into `force_x` and `force_y` [#83](https://github.com/umami-hep/puma/pull/83)
- Adding `puma.PiePlot` class. Pie chart plots with `puma.HistogramPlot` are no longer possible [#70](https://github.com/umami-hep/puma/pull/70)
- Change default labels of `singlebjets` and `singlecjets` [#82](https://github.com/umami-hep/puma/pull/82)
- Support linestyles for variable vs. efficiency plots [#78](https://github.com/umami-hep/puma/pull/78)

### [v0.1.3]

- Adding more flavours to the global config  [#73](https://github.com/umami-hep/puma/pull/73)
- `ratio_group` in `puma.Histogram` objects can no longer be set via `flavour` argument [#74](https://github.com/umami-hep/puma/pull/74)
- Adding example for `plt.show` replacement + adding theme switcher button to docs [#72](https://github.com/umami-hep/puma/pull/72)
- Adding `atlas_tag_outside` and change default for `atlas_second_tag` [#71](https://github.com/umami-hep/puma/pull/71)
- Change default *bb*-jets colour to dark red and vlines to black [#69](https://github.com/umami-hep/puma/pull/69)
- Adding more general `ratio_group` argument to `puma.Histogram` [#67](https://github.com/umami-hep/puma/pull/67)
- Adding `calc_separation()` to `puma.metrics`, which allows to calculate the separation between two distributions [#27](https://github.com/umami-hep/puma/pull/27)
- Adding Zenodo link

### [v0.1.2]

- Adding automated coverage comment for pull request [#58](https://github.com/umami-hep/puma/pull/58)
- Fix that colour and legend label can be individually modified in case of flavoured histogram [#57](https://github.com/umami-hep/puma/pull/57)

### [v0.1.1]

- Adding documentation for updating the version switcher in the docs [#49](https://github.com/umami-hep/puma/pull/49)
- Adding version support in docs [#42](https://github.com/umami-hep/puma/pull/42)[#45](https://github.com/umami-hep/puma/pull/45)
- Adding development guidelines to the docs [#41](https://github.com/umami-hep/puma/pull/41)
- Adding `logx` [#40](https://github.com/umami-hep/puma/pull/40)
- Adding example page for the fraction scan plots [#38](https://github.com/umami-hep/puma/pull/38)
- Add warning for plotting normalised histograms without saying to in the ylabel [#34](https://github.com/umami-hep/puma/pull/34)
- Adding Docker images [#33](https://github.com/umami-hep/puma/pull/33)
- Remove warning when adding more than one reference histogram is case is allowed [#32](https://github.com/umami-hep/puma/pull/32)
- Update documentation [#24](https://github.com/umami-hep/puma/pull/24)

### [v0.1.0]

- Set default number of ratio panels to 0 [#17](https://github.com/umami-hep/puma/pull/17)
- Adding uncertainties to Roc main plots and improving dummy data generator [#14](https://github.com/umami-hep/puma/pull/14)
- Fixing all PEP8 related variable namings [#13](https://github.com/umami-hep/puma/pull/13)
- Adding FractionScan to puma [#8](https://github.com/umami-hep/puma/pull/8)
