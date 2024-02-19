
# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

# Standard library
import logging  # noqa: E402

logging.basicConfig()
logger = logging.getLogger("ultrasatutils")

from .utils import get_transit_SNR, magnitude_to_ULTRASAT