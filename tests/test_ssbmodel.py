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


def test_statstruk_ssbmodel_verbose() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)

    mod1 = ratemodel(p_data, s_data, id_nr="id")
    assert mod1.verbose == 1
    mod1.change_verbose(2)
    assert mod1.verbose == 2


def test_statruk_ssbmodel_check() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)

    # check error raised if wrong variable name is specified
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


def test_statruk_ssbmodel_check_Int64() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)

    # Check that numeric columns of pandas type Int64 are ok
    s_data.employees = s_data.employees.astype("Int64")
    mod1 = ratemodel(p_data, s_data, id_nr="id")


def test_statstruk_ssbmodel_duplicates() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)
    s_data.iloc[0, 0] = 9

    with pytest.raises(ValueError) as exc_info:
        mod1 = ratemodel(p_data, s_data, id_nr="id")
    assert (
        exc_info.value.args[0]
        == "Duplicates found in sample_data based on id. Please fix before proceeding."
    )
