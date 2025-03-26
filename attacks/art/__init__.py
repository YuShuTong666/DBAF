"""
The Adversarial Robustness Toolbox (ART).
"""
import logging.config

# Project Imports
from attacks.art import attacks
# from attacks.art import defences
from attacks.art import estimators
from attacks.art import evaluations
from attacks.art import metrics
from attacks.art import preprocessing

# Semantic Version
__version__ = "1.11.0"

# pylint: disable=C0103

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.NullHandler",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "art": {
            "handlers": ["default"]
        },
        "tests": {
            "handlers": ["test"],
            "level": "INFO",
            "propagate": True
        },
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)
