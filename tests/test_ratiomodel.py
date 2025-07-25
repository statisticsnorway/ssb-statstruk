# Tests for ratio models
# mypy: ignore-errors

# Tests to add:
# - Check for message for rstud in 2 obs strata

# Import libraries
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from statstruk import RatioModel

# Read in test data
sample_file = Path(__file__).parent / "data" / "sample_data.csv"
pop_file = Path(__file__).parent / "data" / "pop_data.csv"

sample1_file = Path(__file__).parent / "data" / "sample_data_1obs.csv"
pop1_file = Path(__file__).parent / "data" / "pop_data_1obs.csv"

s_data = pd.read_csv(sample_file)
p_data = pd.read_csv(pop_file)

# Add country variable
s_data["country"] = 1
p_data["country"] = 1


def test_statstruk_ratiomodel() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
    assert isinstance(mod1.get_coeffs, pd.DataFrame)


def test_statstruk_ratiomodel_nostrata() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", control_extremes=False)
    assert mod1.get_coeffs.shape[0] == 1


def test_statstruk_ratiomodel_liststrata() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var=["industry", "size"],
        control_extremes=False,
    )
    assert mod1.get_coeffs.shape[0] == 15


def test_statstruk_ratiomodel_excludes() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees", y_var="job_vacancies", strata_var="industry", exclude=[5, 9]
    )
    assert mod1.get_coeffs["_strata_var_mod"].iloc[2] == "B_surprise_9"
    assert mod1.get_estimates().shape[0] == 5
    assert mod1.get_weights().estimation_weights.iloc[0] == 1


def test_statstruk_ratiomodel_excludes_missing() -> None:
    """Observation 9855 is missing y-value and is removed. It should therefore be ignored when exclude is specified"""
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        exclude=[9855, 9912],
        control_extremes=False,
    )
    assert isinstance(mod1.get_coeffs, pd.DataFrame)


def test_statstruk_ratiomodel_get_estimates() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
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
    assert int(est_df["job_vacancies_EST"].iloc[0]) == 24186  # check this with struktur

    # test for cross strata domain estimates
    est_df2 = mod1.get_estimates("size")
    assert int(est_df2["job_vacancies_EST"].iloc[0]) == 818  # check this with struktur

    # test for aggregated domain estimates
    est_df3 = mod1.get_estimates("country")
    assert (
        int(est_df3["job_vacancies_EST"].values[0]) == 119773
    )  # check this with struktur


def test_statstruk_ratiomodel_uncertainty_type() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )
    cv_df = mod1.get_estimates(uncertainty_type="CV", variance_type="standard")
    columns_ending_with_cv = [
        column for column in cv_df.columns if column.endswith("_CV")
    ]
    assert len(columns_ending_with_cv) > 0
    other_df = mod1.get_estimates(
        uncertainty_type="CI_SE_VAR", variance_type="standard"
    )
    columns_ending_with_lb = [
        column for column in other_df.columns if column.endswith("_LB")
    ]
    assert len(columns_ending_with_lb) > 0
    columns_ending_with_cv = [
        column for column in other_df.columns if column.endswith("_CV")
    ]
    assert len(columns_ending_with_cv) == 0


def test_statstruk_ratiomodel_get_extremes() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
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
    ex_df4 = mod1.get_extremes(rbound=3, threshold_type="rstud")
    assert ex_df4.shape[0] == 1

    mod2 = RatioModel(p_data, s_data, id_nr="id")
    mod2.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )
    with pytest.raises(RuntimeError):
        mod2.get_extremes()


def test_statstruk_ratiomodel_standard() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
    out = mod1.get_estimates(variance_type="standard")
    assert np.isclose(np.round(out["job_vacancies_CV"].iloc[0], 4), 3.9762, rtol=1e-09, atol=1e-09)


def test_statstruk_ratiomodel_nocontrol() -> None:
    mod1 = RatioModel(p_data, s_data, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )
    out = mod1.get_estimates(variance_type="standard")
    assert np.isclose(np.round(out["job_vacancies_CV"].iloc[0], 4), 3.9762, rtol=1e-09, atol=1e-09)
    with pytest.raises(RuntimeError):
        mod1.get_extremes()


def test_statstruk_ratiomodel_1obs(caplog):
    sample1 = pd.read_csv(sample1_file)
    pop1 = pd.read_csv(pop1_file)
    sample1 = sample1.loc[sample1.job_vacancies.notna(), :]
    mod1 = RatioModel(pop1, sample1, id_nr="id", logger_level="info")

    with caplog.at_level(logging.INFO, logger="model"):
        mod1.fit(
            x_var="employees",
            y_var="job_vacancies",
            strata_var="industry",
            control_extremes=False,
        )
        msg_one = "Stratum: 'G', has only one observation and has 0 variance."
        assert msg_one in caplog.text


def test_statstruk_ratiomodel_check_neg() -> None:
    s_data_new = pd.read_csv(sample_file)
    p_data = pd.read_csv(pop_file)
    s_data_new.iloc[0, 1] = -5

    # Check that negetive values raise error
    with pytest.raises(ValueError) as error_mes:
        mod1 = RatioModel(p_data, s_data_new, id_nr="id")
        mod1.fit(
            x_var="employees",
            y_var="job_vacancies",
            strata_var="industry",
            control_extremes=False,
        )
    assert (
        "There are negative values in the variable 'employees' in the sample dataset. Consider a log transformation or another type of model."
        in str(error_mes.value)
    )


def test_statstruk_ratiomodel_check_sample0(caplog) -> None:
    s_data_new = pd.read_csv(sample_file)
    s_data_new = s_data_new.loc[s_data_new.job_vacancies.notna(), :]

    # create a x = 0
    s_data_new.iloc[0, 1] = 0

    # Check that a message is raised
    mod1 = RatioModel(p_data, s_data_new, id_nr="id", logger_level="info")

    with caplog.at_level(logging.INFO, logger="model"):
        mod1.fit(
            x_var="employees",
            y_var="job_vacancies",
            strata_var="industry",
            control_extremes=True,
        )
        msg = "Zero counts were recorded in the x variable in stratum 'B' and were adjusted to 1 for calculations. Extreme controls will not be done for these observations.\n"
        assert msg in caplog.text

    # Check values for rstud and G are na
    assert np.isnan(mod1.get_obs["B"]["G"])[0]


def test_statstruk_ratiomodel_check_pop0() -> None:
    sample1 = pd.read_csv(sample1_file)
    pop1 = pd.read_csv(pop1_file)

    mod1 = RatioModel(pop1, sample1, id_nr="id")
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )

    # check those with x=0 in population are imputed with zero
    imp = mod1.get_imputed()
    assert imp["job_vacancies_imp"].iloc[0] == 0


def test_stastruk_ratiomodel_check_allbutone0(caplog) -> None:
    sample1 = pd.read_csv(sample1_file)
    sample1 = sample1.loc[sample1.job_vacancies.notna(), :]
    pop1 = pd.read_csv(pop1_file)
    new_rows = pd.DataFrame(
        {
            "id": [10006, 10007],
            "employees": [50, 50],
            "employees_f": [20, 25],
            "employees_m": [30, 25],
            "turnover": [15000, 15000],
            "size": ["mid", "mid"],
            "industry": ["G", "G"],
        }
    )
    pop1 = pd.concat([pop1, new_rows], ignore_index=True)

    new_rows2 = pd.DataFrame(
        {
            "id": [10006, 10007],
            "employees": [50, 50],
            "employees_f": [20, 25],
            "employees_m": [30, 25],
            "turnover": [15000, 15000],
            "size": ["mid", "mid"],
            "industry": ["G", "G"],
            "job_vacancies": [0, 0],
            "sick_days": [70, 65],
            "sick_days_f": [35, 25],
            "sick_days_m": [35, 40],
        }
    )
    sample1 = pd.concat([sample1, new_rows2], ignore_index=True)

    mod1 = RatioModel(pop1, sample1, id_nr="id", logger_level="warning")

    with caplog.at_level(logging.WARNING, logger="model"):
        mod1.fit(
            x_var="employees",
            y_var="job_vacancies",
            strata_var="industry",
            control_extremes=True,
        )
        msg_log = "Only one non-zero value found for 'job_vacancies' in strata: ['G']."
        assert msg_log in caplog.text

    # Check that obs are given na
    assert np.isnan(mod1.get_obs["G"]["G"])[0]


def test_stastruk_ratiomodel_check_all0(caplog) -> None:
    sample1 = pd.read_csv(sample1_file)
    sample1 = sample1.loc[sample1.job_vacancies.notna(), :]
    pop1 = pd.read_csv(pop1_file)
    new_rows = pd.DataFrame(
        {
            "id": [10006, 10007],
            "employees": [50, 50],
            "employees_f": [20, 25],
            "employees_m": [30, 25],
            "turnover": [15000, 15000],
            "size": ["mid", "mid"],
            "industry": ["G", "G"],
        }
    )
    pop1 = pd.concat([pop1, new_rows], ignore_index=True)

    new_rows2 = pd.DataFrame(
        {
            "id": [10006, 10007],
            "employees": [50, 50],
            "employees_f": [20, 25],
            "employees_m": [30, 25],
            "turnover": [15000, 15000],
            "size": ["mid", "mid"],
            "industry": ["G", "G"],
            "job_vacancies": [0, 0],
            "sick_days": [70, 65],
            "sick_days_f": [35, 25],
            "sick_days_m": [35, 40],
        }
    )
    sample1 = pd.concat([sample1, new_rows2], ignore_index=True)
    sample1.loc[sample1.id == 10001, "job_vacancies"] = 0

    mod1 = RatioModel(pop1, sample1, id_nr="id", logger_level="info")

    with caplog.at_level(logging.INFO, logger="model"):
        mod1.fit(
            x_var="employees",
            y_var="job_vacancies",
            strata_var="industry",
            control_extremes=True,
        )
        msg_ex = "All values for 'job_vacancies' in stratum 'G' were zero. Extreme values need to be checked in other ways for this stratum."
        assert msg_ex in caplog.text

    # Check that obs are given na
    assert all(np.isnan(mod1.get_obs["G"]["G"]))


def test_stastruk_ratiomodel_negative_variance() -> None:
    sample1 = pd.read_csv(sample_file)
    sample1 = sample1.loc[sample1.job_vacancies.notna(), :]
    pop1 = pd.read_csv(pop_file)

    # Create more in sample than population
    sample_ids = [8002, 8017, 8022, 8064, 8070]
    pop_ids = [8002, 8017, 8022]

    sample1 = sample1.loc[(sample1.industry != "F") | (sample1.id.isin(sample_ids))]
    pop1 = pop1.loc[(pop1.industry != "F") | (pop1.id.isin(pop_ids))]

    mod1 = RatioModel(pop1, sample1, id_nr="id", verbose=2)
    mod1.fit(
        x_var="employees",
        y_var="job_vacancies",
        strata_var="industry",
        control_extremes=False,
    )
    res = mod1.get_estimates()

    # Check that obs are given na
    assert res["job_vacancies_CV2"].iloc[4] == 0
