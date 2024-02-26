"""Configuration for logger of puma."""

from __future__ import annotations

import logging
import os
from typing import ClassVar


class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colours and count warning / errors using implementation
    from
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output.
    """

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    debugformat = "%(asctime)s - %(levelname)s:%(name)s: %(message)s (%(filename)s:%(lineno)d)"
    date_format = "%(levelname)s:%(name)s: %(message)s"

    formats: ClassVar = {
        logging.DEBUG: grey + debugformat + reset,
        logging.INFO: green + date_format + reset,
        logging.WARNING: yellow + date_format + reset,
        logging.ERROR: red + debugformat + reset,
        logging.CRITICAL: bold_red + debugformat + reset,
    }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_log_level(
    level: str,
):
    """Get logging levels with string key.

    Parameters
    ----------
    level : str
        Log level as string.

    Returns
    -------
    logging level
        logging object with log level info

    Raises
    ------
    ValueError
        If non-valid option is given
    """
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    if level not in log_levels:
        raise ValueError(f"The 'DebugLevel' option {level} is not valid.")
    return log_levels[level]


def initialise_logger(
    log_level: str | None = None,
):
    """Initialise.

    Parameters
    ----------
    log_level : str, optional
        Logging level defining the verbose level. Accepted values are:
        CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET, by default None
        If the log_level is not set, it will

    Returns
    -------
    logger
        logger object with new level set
    """
    log_level = get_log_level(
        os.environ.get("LOG_LEVEL", "INFO") if log_level is None else log_level
    )

    puma_logger = logging.getLogger("puma")
    puma_logger.setLevel(log_level)
    ch_handler = logging.StreamHandler()
    ch_handler.setLevel(log_level)
    ch_handler.setFormatter(CustomFormatter())

    puma_logger.addHandler(ch_handler)
    puma_logger.propagate = False
    return puma_logger


def set_log_level(
    puma_logger,
    log_level: str,
):
    """Setting log level.

    Parameters
    ----------
    puma_logger : logger
        logger object
    log_level : str
        Logging level corresponding CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """
    puma_logger.setLevel(get_log_level(log_level))
    for handler in puma_logger.handlers:
        handler.setLevel(get_log_level(log_level))


logger = initialise_logger()
