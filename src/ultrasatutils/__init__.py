__version__ = "0.0.1"

# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

# Standard library
import logging  # noqa: E402

logging.basicConfig()
logger = logging.getLogger("ultrasatutils")
