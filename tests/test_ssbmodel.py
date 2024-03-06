# General tests
# mypy: ignore-errors

# Import libraries
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statstruk import ratemodel

# Read in test data
sample_file = Path(__file__).parent / "data" / "sample_data.csv"
pop_file = Path(__file__).parent / "data" / "pop_data.csv"

s_data = pd.read_csv(sample_file)
p_data = pd.read_csv(pop_file)

# Add country variable
s_data["country"] = 1
p_data["country"] = 1


def test_statstruk_ssbmodel_verbose() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.change_verbose(2)
    assert mod1.verbose == 2


def test_statruk_ssbmodel_check() -> None:
    with pytest.raises(ValueError):
        mod1 = ratemodel(p_data, s_data, id_nr="virk_nr")
    # check that error raised if missing values in id variable
    s_data.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        mod1 = ratemodel(p_data, s_data, id_nr="id")

    s_data.iloc[0, 0] = 1
    p_data.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        mod1 = ratemodel(p_data, s_data, id_nr="id")
