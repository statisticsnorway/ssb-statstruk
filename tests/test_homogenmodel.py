# Tests for rate models
# mypy: ignore-errors

# Tests to add:

# Import libraries
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from statstruk import homogenmodel

# Read in test data
sample_file = Path(__file__).parent / "data" / "sample_data.csv"
pop_file = Path(__file__).parent / "data" / "pop_data.csv"

sample1_file = Path(__file__).parent / "data" / "sample_data_1obs.csv"
pop1_file = Path(__file__).parent / "data" / "pop_data_1obs.csv"


def test_statstruk_homogenmodel() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)
    mod = homogenmodel(p_data, s_data, id_nr="id")
    mod.fit(y_var="job_vacancies", strata_var="industry", control_extremes=False)
    assert isinstance(mod.get_coeffs, pd.DataFrame)


def test_statstruk_homogenmodel_nostrata() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)
    mod = homogenmodel(p_data, s_data, id_nr="id")
    mod.fit(y_var="job_vacancies", control_extremes=False)
    assert mod.get_coeffs.shape[0] == 1


def test_statstruk_homogenmodel_liststrata() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)
    mod = homogenmodel(p_data, s_data, id_nr="id")
    mod.fit(
        y_var="job_vacancies",
        strata_var=["industry", "size"],
        control_extremes=False,
    )
    assert mod.get_coeffs.shape[0] == 15


def test_statstruk_homogenmodel_excludes() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)
    mod = homogenmodel(p_data, s_data, id_nr="id")
    mod.fit(
        y_var="job_vacancies",
        strata_var="industry",
        exclude=[5, 9],
        control_extremes=False,
    )
    assert mod.get_coeffs.iloc[1, 0] == "B_surprise_5"
    assert mod.get_estimates().shape[0] == 5
    assert mod.get_weights().estimation_weights.iloc[0] == 1


def test_statatruk_homogenmodel_weights() -> None:
    s_data = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)
    mod = homogenmodel(p_data, s_data, id_nr="id")
    mod.fit(y_var="job_vacancies", strata_var="industry", control_extremes=False)
    N = mod.get_estimates().iloc[0, 0]
    n = mod.get_estimates().iloc[0, 1]
    assert mod.get_weights().loc[0, "estimation_weights"] == N / n
    assert mod.get_weights().loc[:, "estimation_weights"].sum() == 10000
