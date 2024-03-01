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

# +
# sample_df.head()
# -


# ## Standard run of rate model

mod1 = ratemodel(pop_df, sample_df, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)

mod1.get_estimates()

mod1.get_estimates("size")

mod1.get_estimates("country")

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
