# Code for importing library and used for working and testing code interactively
import pandas as pd
from statstruk import ratemodel
import pytest


# Read in test data
sample_file = "../tests/data/sample_data.csv"
pop_file = "../tests/data/pop_data.csv"

sample1_file = "../tests/data/sample_data_1obs.csv"
pop1_file = "../tests/data/pop_data_1obs.csv"

s_data = pd.read_csv(sample_file)
p_data = pd.read_csv(pop_file)

sample1 = pd.read_csv(sample1_file)
pop1 = pd.read_csv(pop1_file)
# -


mod1 = ratemodel(pop1, sample1, id_nr="id")
mod1.fit(
    x_var="employees",
    y_var="job_vacancies",
    strata_var="industry",
    control_extremes=True,
)
