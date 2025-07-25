# Homogen model class

from typing import Any

import pandas as pd
import statsmodels.formula.api as smf  # type: ignore

from .stratifiedmodel import StratifiedModel


class HomogenModel(StratifiedModel):
    """Class for estimating statistics for business surveys using a homogeneous model."""

    def fit(
        self,
        y_var: str,
        strata_var: str | list[str] = "",
        exclude: list[str | int] | None = None,
        remove_missing: bool = True,
    ) -> None:
        """Run and fit a homogeneous model within strata.

        Args:
            y_var: The target variable to estimate from the survey.
            strata_var: The stratification variable.
            exclude: List of ID numbers for observations to exclude.
            remove_missing: Whether to automatically remove units in the sample that are missing x or y values.
        """
        # Fit the model
        super()._fit(
            y_var=y_var,
            x_var="",
            strata_var=strata_var,
            control_extremes=False,
            exclude=exclude,
            remove_missing=remove_missing,
            method="homogen",
            method_function=self._homogen_method,
            ai_function=self._get_ai_homogen,
        )

    def _homogen_method(
        self,
        stratum: str,
        group: pd.DataFrame,
        stratum_info: dict[str, Any],
        obs_info: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        formula = self.y_var + "~ 1"

        # Fit model
        model = smf.ols(formula, data=group).fit()
        sigma2 = model.scale
        stratum_info.update(
            {
                f"{self.y_var}_beta": model.params["Intercept"].item(),
                "sigma2": sigma2,
            }
        )

        influence = model.get_influence()
        hats = influence.hat_matrix_diag
        obs_info.update(
            {
                "resids": model.resid.values,
                "hat": hats,
            }
        )

        return stratum_info, obs_info

    def _get_ai_homogen(self, strata: str) -> Any:
        """Get ai values for robust variance for homogen model."""
        n = self.strata_results[strata]["n"]
        nr = self.strata_results[strata]["N"] - n
        return nr / n
