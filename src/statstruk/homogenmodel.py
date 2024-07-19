# Homogen model

from typing import Any

import pandas as pd
import statsmodels.formula.api as smf  # type: ignore

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

    def fit(
        self,
        y_var: str,
        strata_var: str | list[str] = "",
        control_extremes: bool = False,
        exclude: list[str | int] | None = None,
        remove_missing: bool = True,
        rbound: float = 2,
        gbound: float = 2,
    ) -> None:
        """Run and fit a homogeneous model within strata.

        Args:
            y_var: The target variable to estimate from the survey.
            strata_var: The stratification variable.
            control_extremes: Whether the model should be fitted in a way that allows for extremes value controls. (Not implemented)
            exclude: List of ID numbers for observations to exclude.
            remove_missing: Whether to automatically remove units in the sample that are missing x or y values.
            rbound: Multiplicative value to determine the extremity of the studentized residual values.
            gbound: Multiplicative value to determine the extremity of the G values.
        """
        if control_extremes:
            print(
                "Extreme control not implemented for homogen model. Running model without controls"
            )
            control_extremes = False

        super()._fit(
            y_var=y_var,
            x_var="",
            strata_var=strata_var,
            control_extremes=control_extremes,
            exclude=exclude,
            remove_missing=remove_missing,
            rbound=rbound,
            gbound=gbound,
            method="homogen",
            method_function=self._homogen_method,
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

        # NOT implemented
        # if self.control_extremes:
        #    if len(group) == 2:
        #        print(
        #            f"Extreme values not able to be detected in stratum: {stratum!r} due to too few observations."
        #        )
        #        obs_info.update({"rstud": np.nan, "G": np.nan, f"{self.y_var}_beta_ex": np.nan})
        #    else:
        #        rstuds = model.get_influence().get_resid_studentized_external()
        #        obs_info.update({"resids": model.resid.values, "hat": hats,"rstud": rstuds,
        #                         "G": rstuds * (np.sqrt(hats / (1.0 - hats))), f"{self.y_var}_beta_ex": np.nan})

        return stratum_info, obs_info

    def get_estimates(
        self,
        domain: str = "",
        uncertainty_type: str = "CV",
        variance_type: str = "robust",
        return_type: str = "unbiased",
        output_type: str = "table",
    ) -> pd.DataFrame:
        """Get estimates for previously run model within strata or domains. Variance and CV estimates are returned for each domain.

        Args:
            domain: Name of the variable to use for estimation. Should be in the population data.
            uncertainty_type: Which uncertainty measures to return. Choose between 'CV' (default) for coefficient of variation, 'VAR' for variance, 'SE' for standard errors, 'CI' for confidence intervals. Multiple measures can be returned with combinations of these, for example "CV_SE" returns both the coefficient of variation and the standard error.
            variance_type: Choose from 'robust' or 'standard' estimation of variance. Currently only robust estimation is calculated for strata and aggregated strata domains estimation and standard for other domains.
            return_type: String for which robust estimates to return. Choose 'unbiased' to return only the unbiased robust variance estimate or 'all' to return all three.
            output_type: String for output type to return. Default 'table' returns a table with estimates per strata or domain group. Alternatively choose 'weights' to return the sample file with weights and estimates or 'imputed' to return a population file with mass imputed values and estimates.

        Returns:
            A pd.Dataframe is returned conatining estimates and variance/coefficient of variation estimations for each domain.

        """
        return super()._get_estimates(
            domain=domain,
            uncertainty_type=uncertainty_type,
            variance_type=variance_type,
            return_type=return_type,
            output_type=output_type,
            ai_function=self._get_ai_homogen,
        )

    def _get_ai_homogen(self, strata: str) -> Any:
        """Get ai values for robust variance for homogen model."""
        n = self.strata_results[strata]["n"]
        Nr = self.strata_results[strata]["N"] - n
        return Nr / n
