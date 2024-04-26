"""Lalonde dataset (experimental and observational).

See also here for details
https://rugg2.github.io/Lalonde%20dataset%20-%20Causal%20Inference.html
"""

from typing import Dict
import os
import pandas as pd

from . import base


_BASE_URL = "http://www.nber.org/~rdehejia/data"

_COLS = [
    "treatment",
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "earnings1974",
    "earnings1975",
    "earnings1978",
]


_OUTCOME = "earnings1978"
_TREATMENT = "treatment"
_FEATURES = list(set(_COLS) - set([_OUTCOME, _TREATMENT]))

_FILENAMES = {
    "experimental_control": "nswre74_control.txt",
    "experimental_treatment": "nswre74_treated.txt",
    "observational_control": "cps_controls.txt",
}

_URL_FILEPATHS = {k: os.path.join(_BASE_URL, v) for k, v in _FILENAMES.items()}


_EXPERIMENTAL_CONTROL = None
_EXPERIMENTAL_TREATMENT = None
_OBSERVATIONAL_CONTROL = None


def _read_file(txt_file: str) -> pd.DataFrame:
    print("Reading data from %s" % txt_file)
    return pd.read_csv(txt_file, delim_whitespace=True, header=None, names=_COLS)


def _get_all_data() -> Dict[str, pd.DataFrame]:
    """Gets all data as a dictionary.  Only loads from URL once."""
    global _EXPERIMENTAL_CONTROL
    if _EXPERIMENTAL_CONTROL is None:
        _EXPERIMENTAL_CONTROL = _read_file(_URL_FILEPATHS["experimental_control"])

    global _EXPERIMENTAL_TREATMENT
    if _EXPERIMENTAL_TREATMENT is None:
        _EXPERIMENTAL_TREATMENT = _read_file(_URL_FILEPATHS["experimental_treatment"])

    global _OBSERVATIONAL_CONTROL
    if _OBSERVATIONAL_CONTROL is None:
        _OBSERVATIONAL_CONTROL = _read_file(_URL_FILEPATHS["observational_control"])

    return {
        "experimental_control": _EXPERIMENTAL_CONTROL,
        "experimental_treatment": _EXPERIMENTAL_TREATMENT,
        "observational_control": _OBSERVATIONAL_CONTROL,
    }


class Lalonde(base.CausalDataset):
    """Get the Lalonde data as a CausalDataset.

    This will read data from external data source at
    http://www.nber.org/~rdehejia/data.
    """

    def __init__(
        self,
        observational_control: bool = False,
        add_is_unemployed_features: bool = False,
    ):
        """Initializes the class.

        Args:
          observational_control: should control be observational or experimental.
        """
        all_data = _get_all_data()
        if observational_control:
            comb_df = pd.concat(
                [all_data["experimental_treatment"], all_data["observational_control"]],
                axis=0,
            )
        else:
            comb_df = pd.concat(
                [all_data["experimental_treatment"], all_data["experimental_control"]],
                axis=0,
            )

        all_features = _FEATURES
        if add_is_unemployed_features:
            for y in ["1974", "1975"]:
                feat_name = "is_unemployed_" + y
                comb_df[feat_name] = (comb_df["earnings" + y] <= 1.0).astype(float)
                all_features.append(feat_name)

        super().__init__(
            outcomes=comb_df[_OUTCOME],
            treatments=comb_df[_TREATMENT],
            features=comb_df[all_features],
        )
        self._observational_control = observational_control
        self._add_is_unemployed_features = add_is_unemployed_features
