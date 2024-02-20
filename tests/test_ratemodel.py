# Tests for rate models

# Import libraries
from importlib.resources import path

import pandas as pd
import pytest

from statstrukt import ratemodel

# read in data
s_data = pd.read_csv(path("statstrukt.data", "sample_data.csv"))
p_data = pd.read_csv(path("statstrukt.data", "pop_data.csv"))

# Add country variable
s_data["country"] = 1
p_data["country"] = 1


def test_statstrukt_ratemodel():
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
    assert isinstance(mod1.get_coeffs, pd.DataFrame)


def teststatstrukt_ratemodel_excludes():
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees", y_var="job_vacancies", strata_var="industry", exclude=[5, 9]
    )
    assert mod1.get_coeffs._strata_var_mod[2] == "B_surprise_9"
    assert mod1.get_estimates().shape[0] == 5
    assert mod1.get_weights().estimation_weights[0] == 1


def test_statstrukt_ratemodel_get_estimates():
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    with pytest.raises(RuntimeError):
        mod1.get_estimates()
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")

    # test estimates can be extracted
    est_df = mod1.get_estimates()
    assert int(est_df["job_vacancies_est"][0]) == 24186  # check this with struktur

    # test for cross strata domain estimates
    est_df2 = mod1.get_estimates("size")
    assert int(est_df2["job_vacancies_est"][0]) == 818  # check this with struktur

    # test for aggregated domain estimates
    est_df3 = mod1.get_estimates("country")
    assert (
        int(est_df3["job_vacancies_est"].values[0]) == 119773
    )  # check this with struktur


def test_statstrukt_ratemodel_get_extremes():
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
