"""
Unit test script for the functions in utils.logging
"""

import unittest

from puma.utils import logger, set_log_level
from puma.utils.logging import get_log_level

set_log_level(logger, "DEBUG")


class get_log_level_TestCase(unittest.TestCase):
    """Test class for the puma logger"""

    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            get_log_level("TEST")
