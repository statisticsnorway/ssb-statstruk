# # Code for running and testing ratemodel

# +
# mypy: ignore-errors

# +
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf

import numpy as np
import pandas as pd
import random
import importlib
import dapla as dp

# -

from statstruk import ratemodel

# +
pop_df = pd.read_csv("../tests/data/pop_data.csv")
sample_df = pd.read_csv("../tests/data/sample_data.csv")

pop_df["country"] = 1
sample_df["country"] = 1
# -


# Alternative read from felles on Dapla
bucket = "gs://ssb-prod-dapla-felles-data-delt"
folder = "felles/veiledning/python/eksempler/statstruk"

# +
# dp.write_pandas(df = pop_df,
#                gcs_path = f"{bucket}/{folder}/pop.parquet",
#                file_format = "parquet",)
# dp.write_pandas(df = sample_df,
#                gcs_path = f"{bucket}/{folder}/sample.parquet",
#                file_format = "parquet",)
# -

pop_df = dp.read_pandas(f"{bucket}/{folder}/pop.parquet")
sample_df = dp.read_pandas(f"{bucket}/{folder}/sample.parquet")

# +
# pop_df.head()
# -

sample_df.tail(10)


# ## Standard run of rate model

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)

mod1.get_estimates(uncertainty_type="SE_VAR_CI", variance_type="standard")

mod1.get_estimates("size", uncertainty_type="VAR_CI")

mod1.get_estimates("country", uncertainty_type="CI")

mod1.get_coeffs

mod1.get_extremes(gbound=2, rbound=2)

mod1.get_weights()

mod1.get_imputed()

# ## Test with no strata variable

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(x_var="employees", y_var="job_vacancies")

mod1.get_coeffs.shape

# ## Test with list strata

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var=["industry", "size"],
    rbound=5,
    gbound=5,
)
# mod1.get_extremes()

mod1.get_coeffs.shape

# ## Test with auto remove

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    rbound=5,
    gbound=5,
    exclude_auto=1,
)

# mod1.get_extremes(rbound = 5, gbound = 5)

# ## Standard variance

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
# mod1.get_estimates(variance_type="standard")

# ## No extremes

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=False,
)
# mod1.get_estimates(variance_type="standard")

# ## Model with excludes

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(
    x_var="employees", y_var="job_vacancies", strata_var="industry", exclude=[5, 9]
)

mod1.get_estimates()


mod1.get_extremes()

# ## Set up instance with missing

# should raise an error
sample_df.iloc[0, 0] = np.nan
mod1 = ratemodel(pop_df, sample_df, id_nr="id")

pop_df.iloc[0, 0] = np.nan
mod2 = ratemodel(pop_df, sample_df, id_nr="id")


# ## Tests with data with 1 observation

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")

pop1["country"] = 1
sample1["country"] = 1
# -

sample1.tail(10)

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    exclude=[5, 9],
    control_extremes=False,
)

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    exclude=[9855, 9912],
    control_extremes=False,
)


# ## Test when all x = 0 in one strata

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")

new_rows = pd.DataFrame(
    {
        "id": [10002, 10003],
        "employees": [50, 50],
        "employees_f": [20, 25],
        "employees_m": [30, 25],
        "turnover": [0, 0],
        "size": ["mid", "mid"],
        "industry": ["G", "G"],
    }
)
pop1 = pd.concat([pop1, new_rows], ignore_index=True)

new_rows2 = pd.DataFrame(
    {
        "id": [10002, 10003],
        "employees": [50, 50],
        "employees_f": [20, 25],
        "employees_m": [30, 25],
        "turnover": [0, 0],
        "size": ["mid", "mid"],
        "industry": ["G", "G"],
        "job_vacancies": [35, 45],
        "sick_days": [70, 65],
        "sick_days_f": [35, 25],
        "sick_days_m": [35, 40],
    }
)
sample1 = pd.concat([sample1, new_rows2], ignore_index=True)
sample1.loc[sample1.id == 10001, "turnover"] = 0
# -

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="turnover",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)
mod1.get_obs["G"]

# ## Test when one x=0

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")

new_rows = pd.DataFrame(
    {
        "id": [10005, 10006],
        "employees": [50, 50],
        "employees_f": [20, 25],
        "employees_m": [30, 25],
        "turnover": [1000, 0],
        "size": ["mid", "mid"],
        "industry": ["G", "G"],
    }
)
pop1 = pd.concat([pop1, new_rows], ignore_index=True)

new_rows2 = pd.DataFrame(
    {
        "id": [10005, 10006],
        "employees": [50, 50],
        "employees_f": [20, 25],
        "employees_m": [30, 25],
        "turnover": [1000, 0],
        "size": ["mid", "mid"],
        "industry": ["G", "G"],
        "job_vacancies": [35, 45],
        "sick_days": [70, 65],
        "sick_days_f": [35, 25],
        "sick_days_m": [35, 40],
    }
)
sample1 = pd.concat([sample1, new_rows2], ignore_index=True)

# -

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="turnover",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)
mod1.get_obs["G"]

# ## test when one x=0 in pop

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")


# -

pop1.head(10)

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=False,
)
imp = mod1.get_imputed()
imp["job_vacancies_imputed"][0] == 0


mod1.get_extremes(threshold_type="rstud", rbound=15)

# ## Test when one x is negative

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")

sample1.loc[sample1.id == 5, "turnover"] = -10
pop1.loc[pop1.id == 5, "turnover"] = -10
# -

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="turnover",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)

# ## Test when there is one y=0

pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")
sample1.iloc[0, 7] = 0

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)


# ## Test when all y = 0 in one strata

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")

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
sample1 = sample1.loc[sample1.industry == "G"]
pop1 = pop1.loc[pop1.industry == "G"]
# -

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)


mod1.get_estimates()

mod1.get_obs

# ## Test y=0 for all but one in strata

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")

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

# -

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)

mod1.get_obs["G"]


# ## Test for only 2 in one strata

# +
pop1 = pd.read_csv("../tests/data/pop_data_1obs.csv")
sample1 = pd.read_csv("../tests/data/sample_data_1obs.csv")

new_rows = pd.DataFrame(
    {
        "id": [10006],
        "employees": [45],
        "employees_f": [20],
        "employees_m": [30],
        "turnover": [15000],
        "size": ["mid"],
        "industry": ["G"],
    }
)
pop1 = pd.concat([pop1, new_rows], ignore_index=True)

new_rows2 = pd.DataFrame(
    {
        "id": [10006],
        "employees": [45],
        "employees_f": [20],
        "employees_m": [30],
        "turnover": [15000],
        "size": ["mid"],
        "industry": ["G"],
        "job_vacancies": [0],
        "sick_days": [70],
        "sick_days_f": [35],
        "sick_days_m": [35],
    }
)
sample1 = pd.concat([sample1, new_rows2], ignore_index=True)

mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)
# -

mod1.get_obs["G"]
