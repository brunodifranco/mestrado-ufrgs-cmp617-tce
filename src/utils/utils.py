import json
from typing import Dict
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_json(params_path: Path) -> Dict:
    """
    Cleans data and performs NLP techniques.

    Parameters
    ----------
    params_path : Path
        Path to JSON parameters.

    Returns
    -------
    params_json : Dict
        JSON read in python dict format.
    """

    with open(params_path, "r") as json_file:
        params_json = json.load(json_file)
        return params_json

def pct_format(pct: float, all_vals: int) -> str:
    """
    Calculates percentage and formats as string

    Parameters
    ----------
    pct: float
        Float percentage.
    all_vals: int
        All values

    Returns
    -------
    string
        Formatted percentage.
    """

    absolute = int(pct/100.*np.sum(all_vals))
    return "{:.1f}%\n{:d} Documents".format(pct, absolute)
