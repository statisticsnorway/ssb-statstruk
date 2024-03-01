# Tests for rate models
# mypy: ignore-errors

# Tests to add:
# - Check for message for rstud in 2 obs strata

# Import libraries
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statstrukt import ratemodel

# Read in test data
sample_file = Path(__file__).parent / "data" / "sample_data.csv"
pop_file = Path(__file__).parent / "data" / "pop_data.csv"

s_data = pd.read_csv(sample_file)
p_data = pd.read_csv(pop_file)

# Add country variable
s_data["country"] = 1
p_data["country"] = 1


def test_statstrukt_ratemodel() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
    assert isinstance(mod1.get_coeffs, pd.DataFrame)


def test_statstrukt_ratemodel_nostrata() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", control_extremes=False)
    assert mod1.get_coeffs.shape[0] == 1


def test_statstrukt_ratemodel_liststrata() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var=["industry", "size"],
        control_extremes=False,
    )
    assert mod1.get_coeffs.shape[0] == 15


def test_statstrukt_ratemodel_excludes() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees", y_var="job_vacancies", strata_var="industry", exclude=[5, 9]
    )
    assert mod1.get_coeffs["_strata_var_mod"].iloc[2] == "B_surprise_9"
    assert mod1.get_estimates().shape[0] == 5
    assert mod1.get_weights().estimation_weights[0] == 1


def test_statstrukt_ratemodel_get_estimates() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    with pytest.raises(RuntimeError):
        mod1.get_estimates()
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )

    # test estimates can be extracted
    est_df = mod1.get_estimates()
    assert int(est_df["job_vacancies_est"].iloc[0]) == 24186  # check this with struktur

    # test for cross strata domain estimates
    est_df2 = mod1.get_estimates("size")
    assert int(est_df2["job_vacancies_est"].iloc[0]) == 818  # check this with struktur

    # test for aggregated domain estimates
    est_df3 = mod1.get_estimates("country")
    assert (
        int(est_df3["job_vacancies_est"].values[0]) == 119773
    )  # check this with struktur


def test_statstrukt_ratemodel_get_extremes() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    with pytest.raises(RuntimeError):
        mod1.get_extremes()
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
    ex_df = mod1.get_extremes()
    assert ex_df["id"][0] == 5
    assert ex_df.shape[0] == 111
    ex_df2 = mod1.get_extremes(gbound=10)
    assert ex_df2.shape[0] == 45
    ex_df3 = mod1.get_extremes(rbound=5, gbound=5)
    assert ex_df3.shape[0] == 4

    mod2 = ratemodel(p_data, s_data, id_nr="id")
    mod2.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )
    with pytest.raises(RuntimeError):
        mod2.get_extremes()


def test_statstrukt_ratemodel_auto_extremes() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        rbound=5,
        gbound=5,
        exclude_auto=1,
    )
    ex_df = mod1.get_extremes(rbound=5, gbound=5)
    assert ex_df.shape[0] == 0


def test_statstrukt_ratemodel_standard() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
    out = mod1.get_estimates(variance_type="standard")
    assert np.round(out["job_vacancies_CV"].iloc[0], 4) == 3.9762


def test_statstrukt_ratemodel_nocontrol() -> None:
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )
    out = mod1.get_estimates(variance_type="standard")
    assert np.round(out["job_vacancies_CV"].iloc[0], 4) == 3.9762
    with pytest.raises(RuntimeError):
        mod1.get_extremes()
