"""Unit test script for the functions in utils.logging."""

from __future__ import annotations

import unittest

from puma.utils import logger, set_log_level
from puma.utils.logging import get_log_level

set_log_level(logger, "DEBUG")


class LogLevelTestCase(unittest.TestCase):
    """Test class for the puma logger."""

    def test_wrong_input(self):
        """Test scenario when a wrong log level is being provided."""
        with self.assertRaises(ValueError):
            get_log_level("TEST")
