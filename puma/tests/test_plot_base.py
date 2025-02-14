"""Unit test script for the functions in plot_base.py."""

from __future__ import annotations

import unittest
from unittest.mock import ANY, MagicMock, patch

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from puma.plot_base import PlotBase, PlotObject
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class PlotObjectTestCase(unittest.TestCase):
    """Test class for the puma.PlotObject dataclass."""

    def test_only_one_input_figsize(self):
        """Test only one input figsize."""
        with self.assertRaises(ValueError):
            PlotObject(figsize=1)

    def test_only_tuple_three_inputs_figsize(self):
        """Test only tuple three inputs figsize."""
        with self.assertRaises(ValueError):
            PlotObject(figsize=(1, 2, 3))

    def test_tuple_input_figsize(self):
        """Test tuple input figsize."""
        figsize = PlotObject(figsize=(1, 2)).figsize
        self.assertEqual(figsize, (1, 2))

    def test_list_input_figsize(self):
        """Test list input figsize."""
        figsize = PlotObject(figsize=[1, 2]).figsize
        self.assertEqual(figsize, (1, 2))

    def test_list_input_wrong_len_figsize(self):
        """Test list input wrong len figsize."""
        with self.assertRaises(ValueError):
            PlotObject(figsize=[1, 2, 3])

    def test_wrong_n_ratio_panels(self):
        """Test wrong n ratio panels."""
        with self.assertRaises(ValueError):
            PlotObject(n_ratio_panels=5)

    def test_wrong_ymin_ratio(self):
        """Test wrong ymin_ratio."""
        with self.assertRaises(ValueError):
            PlotObject(n_ratio_panels=2, ymin_ratio=[0])

    def test_wrong_ymax_ratio(self):
        """Test wrong ymax_ratio."""
        with self.assertRaises(ValueError):
            PlotObject(n_ratio_panels=2, ymax_ratio=[0])

    def test_set_ratio_label_invalid_ratio_panel(self):
        plot_object = PlotBase(n_ratio_panels=2)
        plot_object.set_ratio_label(ratio_panel=1, label="Label")
        plot_object.set_ratio_label(ratio_panel=2, label="Label")
        with self.assertRaises(ValueError):
            plot_object.set_ratio_label(ratio_panel=3, label="Label")


class PlotBaseTestCase(unittest.TestCase):
    """Test class for the puma.PlotBase class."""

    def test_ymin_ymax_ratio(self):
        """Test correct ymax/ymin_ratio."""
        plot_object = PlotBase(n_ratio_panels=2, ymin_ratio=[0, 1], ymax_ratio=[1, 2])
        plot_object.initialise_figure()
        plot_object.set_y_lim()
        for i in range(2):
            ymin, ymax = plot_object.ratio_axes[i].get_ylim()
            self.assertEqual((ymin, ymax), (i, i + 1))


class TestPlotBase(unittest.TestCase):
    def setUp(self):
        """Create a fresh PlotBase instance before each test."""
        # Default with 0 ratio panels, unless changed in the test.
        self.plot_base = PlotBase()

    def test_default_initialisation(self):
        """Test default initialization of PlotBase."""
        self.assertEqual(self.plot_base.n_ratio_panels, 0)
        self.assertTrue(self.plot_base.logy)
        self.assertFalse(self.plot_base.logx)
        self.assertFalse(self.plot_base.plotting_done)
        self.assertIsNone(self.plot_base.fig)
        self.assertIsNone(self.plot_base.axis_top)
        self.assertEqual(self.plot_base.ratio_axes, [])

    def test_initialise_figure_no_ratio(self):
        """Test initialise_figure() with n_ratio_panels=0."""
        self.plot_base.n_ratio_panels = 0
        self.plot_base.initialise_figure()

        self.assertIsNotNone(self.plot_base.fig)
        self.assertIsInstance(self.plot_base.axis_top, Axes)
        self.assertEqual(len(self.plot_base.ratio_axes), 0)

    def test_initialise_figure_with_one_ratio(self):
        """Test initialise_figure() with n_ratio_panels=1."""
        # Re-create with 1 ratio panel so __post_init__ sets up arrays correctly
        self.plot_base = PlotBase(n_ratio_panels=1)
        self.plot_base.initialise_figure()

        self.assertIsNotNone(self.plot_base.fig)
        self.assertIsInstance(self.plot_base.axis_top, Axes)
        self.assertEqual(len(self.plot_base.ratio_axes), 1)
        self.assertIsInstance(self.plot_base.ratio_axes[0], Axes)

    @patch("puma.plot_base.logger.warning")
    def test_initialise_figure_vertical_split(self, mock_logger):
        """
        Test initialise_figure() with vertical_split=True.
        Since we have n_ratio_panels>0, the code logs a warning instead
        of raising a Python UserWarning.
        """
        self.plot_base.n_ratio_panels = 2
        self.plot_base.vertical_split = True
        self.plot_base.initialise_figure()

        # Because n_ratio_panels>0 but vertical_split=True, we expect a log warning.
        mock_logger.assert_called_once()
        self.assertIsNotNone(self.plot_base.fig)
        self.assertIsNotNone(self.plot_base.axis_top)
        self.assertIsNotNone(self.plot_base.axis_leg)
        # No ratio axes created in vertical split
        self.assertEqual(len(self.plot_base.ratio_axes), 0)

    def test_set_xlim(self):
        """Test set_xlim sets correct x-limits."""
        self.plot_base.initialise_figure()
        self.plot_base.set_xlim(xmin=1, xmax=10)
        self.assertEqual(self.plot_base.axis_top.get_xlim(), (1, 10))

    def test_set_ylim(self):
        """Test set_y_lim sets correct y-limits."""
        self.plot_base.initialise_figure()
        self.plot_base.ymin = 0.5
        self.plot_base.ymax = 100.0
        self.plot_base.set_y_lim()
        ymin, ymax = self.plot_base.axis_top.get_ylim()
        self.assertAlmostEqual(ymin, 0.5)
        self.assertAlmostEqual(ymax, 100.0)

    def test_set_xlabel_no_ratio(self):
        """Test set_xlabel with no ratio panels."""
        self.plot_base.initialise_figure()
        self.plot_base.set_xlabel("My X Label")
        self.assertEqual(self.plot_base.axis_top.get_xlabel(), "My X Label")

    def test_set_xlabel_with_ratio(self):
        """Test set_xlabel with ratio panels -> label on the last ratio axis."""
        self.plot_base = PlotBase(n_ratio_panels=1)
        self.plot_base.initialise_figure()
        self.plot_base.set_xlabel("My X Label")
        self.assertEqual(self.plot_base.ratio_axes[-1].get_xlabel(), "My X Label")

    def test_set_ylabel(self):
        """Test set_ylabel on the top axis."""
        self.plot_base.initialise_figure()
        self.plot_base.set_ylabel(self.plot_base.axis_top, "My Y Label")
        self.assertEqual(self.plot_base.axis_top.get_ylabel(), "My Y Label")

    def test_set_tick_params(self):
        """Test set_tick_params is applied to top and ratio axes."""
        self.plot_base = PlotBase(n_ratio_panels=1)
        self.plot_base.initialise_figure()
        self.plot_base.set_tick_params(labelsize=14)

        top_ticklabels = self.plot_base.axis_top.yaxis.get_ticklabels()
        ratio_ticklabels = self.plot_base.ratio_axes[-1].xaxis.get_ticklabels()
        if top_ticklabels and ratio_ticklabels:
            self.assertEqual(top_ticklabels[0].get_fontsize(), 14)
            self.assertEqual(ratio_ticklabels[0].get_fontsize(), 14)

    def test_draw_vlines(self):
        """Test draw_vlines adds vertical lines to top axis and ratio axes."""
        self.plot_base = PlotBase(n_ratio_panels=1)
        self.plot_base.initialise_figure()
        self.plot_base.draw_vlines(xs=[0.2, 0.3], labels=["20%", "30%"])

        lines_in_top = self.plot_base.axis_top.lines
        lines_in_ratio = self.plot_base.ratio_axes[0].lines
        self.assertEqual(len(lines_in_top), 2)
        self.assertEqual(len(lines_in_ratio), 2)

    @patch("puma.plot_base.PlotBase.is_running_in_jupyter", return_value=True)
    @patch("puma.plot_base.display")  # Patch IPython.display.display
    def test_show_in_jupyter(self, mock_display, mock_jupyter):  # noqa:ARG002
        """
        Test show() in Jupyter environment.
        Expects to call display(self.fig) instead of launching Tkinter.
        """
        self.plot_base.fig = Figure()  # Real figure
        self.plot_base.show()
        mock_display.assert_called_once_with(self.plot_base.fig)

    @patch("puma.plot_base.FigureCanvasTkAgg")
    @patch("puma.plot_base.tk.Tk")
    @patch("puma.plot_base.PlotBase.is_running_in_jupyter", return_value=False)
    def test_show_in_tkinter(self, mock_jupyter, mock_tk, mock_canvas):  # noqa:ARG002
        """
        Test show() in non-Jupyter environment. We'll patch Tk calls
        and FigureCanvasTkAgg to avoid actually opening a GUI or
        running real Tkinter logic that causes KeyError.
        """
        plot_base = PlotBase()
        plot_base.fig = Figure()  # A real Figure is fine, but won't be rendered.

        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_canvas.return_value = MagicMock()

        plot_base.show(auto_close_after=1000)

        # Verify that a Tkinter root window is created
        mock_tk.assert_called_once()

        # Verify the canvas was created with our figure and root
        mock_canvas.assert_called_once_with(plot_base.fig, master=mock_root)

        # Verify that mainloop was called
        mock_root.mainloop.assert_called_once()

        # Verify we scheduled auto-close
        mock_root.after.assert_called_once_with(1000, ANY)

    def test_savefig(self):
        """Test savefig calls figure's savefig() with correct arguments."""
        self.plot_base.initialise_figure()
        with patch.object(self.plot_base.fig, "savefig") as mock_savefig:
            self.plot_base.savefig("test_plot.png", transparent=True, dpi=100)
            mock_savefig.assert_called_once_with(
                "test_plot.png", transparent=True, dpi=100, bbox_inches="tight", pad_inches=0.04
            )

    def test_set_title(self):
        """Test set_title sets the title on the top axis."""
        self.plot_base.initialise_figure()
        self.plot_base.set_title("My Awesome Plot")
        self.assertEqual(self.plot_base.axis_top.get_title(), "My Awesome Plot")

    def test_make_legend(self):
        """Test make_legend creates a legend on the specified axis."""
        self.plot_base.initialise_figure()
        line1 = Line2D([], [], label="Line 1")
        line2 = Line2D([], [], label="Line 2")
        self.plot_base.make_legend(handles=[line1, line2], ax_mpl=self.plot_base.axis_top)

        # Retrieve the legend object:
        legend = self.plot_base.axis_top.get_legend()
        self.assertIsNotNone(legend, "Expected a Legend on axis_top.")
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertListEqual(labels, ["Line 1", "Line 2"])

    def test_make_linestyle_legend(self):
        """Test make_linestyle_legend creates a legend of linestyles."""
        self.plot_base.initialise_figure()
        linestyles = ["-", "--"]
        labels = ["Solid", "Dashed"]
        self.plot_base.make_linestyle_legend(linestyles, labels)

        legend = self.plot_base.axis_top.get_legend()
        self.assertIsNotNone(legend, "Expected a Legend for linestyles.")
        legend_labels = [t.get_text() for t in legend.get_texts()]
        self.assertListEqual(legend_labels, labels)

    def test_set_ratio_label_valid(self):
        """Test setting ratio label on a valid ratio axis."""
        # Create with ratio panels so __post_init__ sets up the arrays:
        self.plot_base = PlotBase(n_ratio_panels=1)
        self.plot_base.initialise_figure()
        self.plot_base.set_ratio_label(1, "Ratio Panel Label")
        self.assertEqual(self.plot_base.ylabel_ratio, ["Ratio Panel Label"])

    def test_set_ratio_label_invalid(self):
        """Test that set_ratio_label raises ValueError if panel index is out of range."""
        self.plot_base = PlotBase(n_ratio_panels=1)
        self.plot_base.initialise_figure()
        with self.assertRaises(ValueError):
            self.plot_base.set_ratio_label(2, "Should fail")

    @patch("puma.plot_base.logger.warning")
    def test_atlasify_not_plotting_done(self, mock_logger):
        """Test atlasify() logs a warning if plotting_done is False and force=False."""
        self.plot_base.initialise_figure()
        self.plot_base.atlasify(force=False)
        # Because plotting_done is still False, we expect a logger.warning:
        mock_logger.assert_called_once()

    @patch("puma.plot_base.logger.warning")
    @patch("atlasify.atlasify")
    def test_atlasify_force(self, mock_atlasify, mock_logger):
        """
        Test atlasify(force=True) applies style even if plotting_done=False.
        Also logs warnings if apply_atlas_style=False or plotting_done=False.
        """
        self.plot_base.initialise_figure()
        self.plot_base.atlasify(force=True)
        # atlasify.atlasify should be called
        mock_atlasify.assert_called()
        # The logger might have warnings about forcing style.
        self.assertTrue(mock_logger.call_count >= 1)

    def test_jupyter_zmq_interactive_shell(self):
        """
        If get_ipython() returns an object with __class__.__name__ == 'ZMQInteractiveShell',
        it should return True (Jupyter Notebook or qtconsole).
        """
        plot_base = PlotBase()
        with patch("puma.plot_base.get_ipython") as mock_get_ipython:
            # Mock object with the right __class__.__name__
            mock_shell = MagicMock()
            mock_shell.__class__.__name__ = "ZMQInteractiveShell"
            mock_get_ipython.return_value = mock_shell

            self.assertTrue(plot_base.is_running_in_jupyter())

    def test_ipython_terminal_shell(self):
        """
        If get_ipython() returns an object with __class__.__name__ == 'TerminalInteractiveShell',
        it should return False (IPython in a terminal, not a Jupyter Notebook).
        """
        plot_base = PlotBase()
        with patch("puma.plot_base.get_ipython") as mock_get_ipython:
            mock_shell = MagicMock()
            mock_shell.__class__.__name__ = "TerminalInteractiveShell"
            mock_get_ipython.return_value = mock_shell

            self.assertFalse(plot_base.is_running_in_jupyter())

    def test_no_ipython(self):
        """
        If get_ipython() returns None, then we assume it's standard Python interpreter,
        so is_running_in_jupyter() should return False.
        """
        plot_base = PlotBase()
        with patch("puma.plot_base.get_ipython") as mock_get_ipython:
            mock_get_ipython.return_value = None

            self.assertFalse(plot_base.is_running_in_jupyter())

    def test_unknown_shell(self):
        """If get_ipython() returns some unknown shell object, we default to False."""
        plot_base = PlotBase()
        with patch("puma.plot_base.get_ipython") as mock_get_ipython:
            mock_shell = MagicMock()
            mock_shell.__class__.__name__ = "SomeRandomShell"
            mock_get_ipython.return_value = mock_shell

            self.assertFalse(plot_base.is_running_in_jupyter())

    def test_import_error(self):
        """
        If there's a NameError or ImportError when calling get_ipython(),
        is_running_in_jupyter() should return False.
        """
        plot_base = PlotBase()
        with patch("puma.plot_base.get_ipython", side_effect=ImportError("No IPython")):
            self.assertFalse(plot_base.is_running_in_jupyter())

    @patch("puma.plot_base.logger.debug")
    def test_close_window_with_valid_root(self, mock_logger):
        """
        Test _close_window when root is not None.
        Should call root.quit(), root.destroy(), and log a debug message.
        """
        plot_base = PlotBase()
        mock_root = MagicMock()

        plot_base.close_window(mock_root)

        mock_root.quit.assert_called_once()
        mock_root.destroy.assert_called_once()
        mock_logger.assert_called_once_with("Closing plot window.")

    @patch("puma.plot_base.logger.debug")
    def test_close_window_with_none(self, mock_logger):
        """
        Test _close_window when root is None.
        Should do nothing and should not log any debug messages.
        """
        plot_base = PlotBase()
        plot_base.close_window(None)

        # root is None, so these should not be called
        mock_logger.assert_not_called()

    def test_show_without_init_figure(self):
        """Test the error when the figure was not initalised."""
        self.plot_base.n_ratio_panels = 0
        with self.assertRaises(ValueError):
            self.plot_base.show()
