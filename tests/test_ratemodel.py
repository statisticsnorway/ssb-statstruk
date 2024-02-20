# Tests for rate models
import unittest
from importlib.resources import path
import pandas as pd
import pytest

from statstrukt import ratemodel

s_data = pd.read_csv(path('statstrukt.data', 'sample_data.csv'))
p_data = pd.read_csv(path('statstrukt.data', 'pop_data.csv'))


def test_strukturmodels_ratemodel():
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    mod1.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
    res1 = mod1.get_estimates()
    assert isinstance(res1, pd.DataFrame)


def test_strukturmodels_getextreme():
    mod1 = ratemodel(p_data, s_data, id_nr="id")
    with pytest.raises(RuntimeError):
        mod1.get_extremes()


if __name__ == '__main__':
    unittest.main()

    
