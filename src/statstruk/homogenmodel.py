# Homogen model

from typing import Any
import numpy as np
import pandas as pd

#import statsmodels.formula.api as smf  # type: ignore

from .stratifiedmodel import stratifiedmodel

class homogenmodel(stratifiedmodel):
    """Class for estimating statistics for business surveys using a homogeneous model."""
    def __init__(
        self,
        pop_data: pd.DataFrame,
        sample_data: pd.DataFrame,
        id_nr: str,
        verbose: int = 1,
    ) -> None:
        """Initialization of object."""
        super().__init__(pop_data, sample_data, id_nr, verbose)    
        
    def fit(self,
        y_var: str,
        strata_var: str | list[str] = "",
        control_extremes: bool = True,
        exclude: list[str | int] | None = None,
        remove_missing: bool = True,
        rbound: float = 2,
        gbound: float = 2
        ) -> None:
        """Run and fit a homogeneous model within strata.

        Args:
            y_var: The target variable to estimate from the survey.
            strata_var: The stratification variable.
            control_extremes: Whether the model should be fitted in a way that allows for extremes value controls.
            exclude: List of ID numbers for observations to exclude.
            remove_missing: Whether to automatically remove units in the sample that are missing x or y values.
            rbound: Multiplicative value to determine the extremity of the studentized residual values.
            gbound: Multiplicative value to determine the extremity of the G values.
        """
        super().fit(
            y_var=y_var,
            x_var="",
            strata_var=strata_var,
            control_extremes=control_extremes,
            exclude=exclude,
            remove_missing=remove_missing,
            rbound=rbound,
            gbound=gbound,
            method = "homogen")
        
        
    

