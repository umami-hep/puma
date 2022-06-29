#!/usr/bin/env python

"""
Unit test script for the functions in utils/histogram.py
"""

# import os
# import tempfile
import unittest

# from puma import Histogram, HistogramPlot
from puma.utils import logger, set_log_level

# import numpy as np


# from puma.utils.histogram import hist_ratio, hist_w_unc

set_log_level(logger, "DEBUG")


class histogram_utils_TestCase(unittest.TestCase):
    """Test class for the puma.utils.histogram functions."""

    pass

    # TODO: Add unit tests for:
    # 1. hist_w_unc (some basic cases)
    # 2. hist_w_unc for weighted calculation
    # 3. hist_ratio
    # 4. Check what happens in hist_w_unc if `bins` (array with bin_edges) AND
    #    `bins_range` is specified (in this case the range should be ignored, since
    #    we just use the np.histogram function and hand the parameters to that function)
