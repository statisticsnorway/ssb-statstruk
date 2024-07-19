# New rate model

# Homogen model

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf  # type: ignore

from .stratifiedmodel import stratifiedmodel


class ratemodel(stratifiedmodel):
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
        x_var: str,
        strata_var: str | list[str] = "",
        control_extremes: bool = True,
        exclude: list[str | int] = [],
        remove_missing: bool = True,
        rbound: float = 2,
        gbound: float = 2,
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
        self._check_variable(x_var, self.pop_data, data_name="population")
        self._check_variable(x_var, self.sample_data, remove_missing=remove_missing)

        self.pop_data[x_var] = self._convert_var(x_var, self.pop_data)
        self.sample_data[x_var] = self._convert_var(x_var, self.sample_data)
        exclude = self._exclude_zeros(
            exclude, self.sample_data, x_var, y_var, self.id_nr
        )

        super()._fit(
            y_var=y_var,
            x_var=x_var,
            strata_var=strata_var,
            control_extremes=control_extremes,
            exclude=exclude,
            remove_missing=remove_missing,
            rbound=rbound,
            gbound=gbound,
            method="rate",
            method_function=self._rate_method,
        )

    def _rate_method(
        self,
        stratum: str,
        group: pd.DataFrame,
        stratum_info: dict[str, Any],
        obs_info: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Rate model method."""
        # stratum_info.update({f"{self.x_var}_sum_sample": group[self.x_var].sum()})
        formula = self.y_var + "~" + self.x_var + "- 1"

        # Create weights
        weights = 1.0 / (group[self.x_var])

        # Fit the weighted least squares model
        model = smf.wls(formula, data=group, weights=weights).fit()
        sigma2 = (1.0 / (len(group) - 1)) * sum(
            model.resid.values**2 / group[self.x_var]
        )

        # Add into stratum information
        stratum_info.update(
            {
                f"{self.y_var}_beta": model.params[
                    self.x_var
                ].item(),  # Series of coefficients
                "sigma2": sigma2,
            }
        )

        # Add in residuals and hat values to observation info
        hats = self._get_hat(group[[self.x_var]].values, weights)
        obs_info.update({"resids": model.resid.values, "hat": hats})

        # Add in studentized residuals and G values if specified
        if self.control_extremes:
            if len(group) == 2:
                print(
                    f"Extreme values not able to be detected in stratum: {stratum!r} due to too few observations."
                )
                obs_info.update(
                    {"rstud": np.nan, "G": np.nan, f"{self.y_var}_beta_ex": np.nan}
                )
            else:
                rstuds = self._get_rstud(
                    y=np.array(group[self.y_var]),
                    res=model.resid.values,
                    x_var=self.x_var,
                    df=group,
                    hh=hats,
                    X=np.array(group[self.x_var]),
                    formula=formula,
                )
                obs_info.update(
                    {
                        "rstud": rstuds[0],
                        "G": rstuds[0] * (np.sqrt(hats / (1.0 - hats))),
                        f"{self.y_var}_beta_ex": rstuds[1],
                    }
                )

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
            ai_function=self._get_ai_rate,
        )

    def _get_ai_rate(self, strata: str) -> pd.Series:  # type: ignore[type-arg]
        """Get ai values for robust variance for rate model."""
        # collect data for strata
        x_pop = self.strata_results[strata][f"{self.x_var}_sum_pop"]
        x_utv = self.strata_results[strata][f"{self.x_var}_sum_sample"]

        # check for x_utv = 0
        if x_utv == 0:
            ai = 0
        # Calculate ai
        else:
            Xr = x_pop - x_utv
            ai = Xr / x_utv

        return ai

    def get_extremes(
        self, threshold_type: str = "both", rbound: float = 2, gbound: float = 2
    ) -> pd.DataFrame:
        """Get observations with extreme values based on their rstudized residual value or G value.

        Args:
            threshold_type: Which threshold type to use. Choose between 'rstud' for studentized residuals, 'G' for dffits/G-value or 'both'(default) for both.
            rbound: Multiplicative value to determine the extremity of the studentized residual values. (Default = 2)
            gbound: Multiplicative value to determine the extremity of the G values. (Default = 2)

        Returns:
            A pd.DataFrame containing units with extreme values beyond a set boundary.
        """
        self._check_extreme_run()

        assert (
            self.method != "homogen"
        ), "Function 'get_extremes' not available for a homogen model"

        # Collect information from strata results and get_obs
        extremes = pd.DataFrame()
        for k in self.get_obs.keys():
            new = pd.DataFrame.from_dict(self.get_obs[k])
            new[f"{self.y_var}_beta"] = self.strata_results[k][f"{self.y_var}_beta"]
            new["_strata_var_mod"] = self.strata_results[k]["_strata_var_mod"]
            new["n"] = self.strata_results[k]["n"]
            new["N"] = self.strata_results[k]["N"]
            new[self.x_var] = new["xvar"]
            new[self.y_var] = new["yvar"]
            new[f"{self.x_var}_sum_pop"] = self.strata_results[k][
                f"{self.x_var}_sum_pop"
            ]
            new[f"{self.y_var}_EST"] = (
                new[f"{self.y_var}_beta"] * new[f"{self.x_var}_sum_pop"]
            )
            new[f"{self.y_var}_EST_ex"] = (
                new[f"{self.y_var}_beta_ex"] * new[f"{self.x_var}_sum_pop"]
            )
            new["gbound"] = gbound * np.sqrt(1 / self.strata_results[k]["n"])
            extremes = pd.concat([extremes, new])

        # create conditions and filter
        condr = (np.abs(extremes["rstud"]) > rbound) & (extremes["rstud"].notna())
        condg = (np.abs(extremes["G"]) > extremes["gbound"]) & (extremes["G"].notna())
        if threshold_type == "rstud":
            extremes = extremes.loc[condr]
        elif threshold_type == "G":
            extremes = extremes.loc[condg]
        else:
            extremes = extremes.loc[condr | condg]

        # Format return object
        extremes = extremes[
            [
                self.id_nr,
                "_strata_var_mod",
                "n",
                "N",
                self.x_var,
                self.y_var,
                f"{self.y_var}_EST",
                f"{self.y_var}_EST_ex",
                "gbound",
                "G",
                "rstud",
            ]
        ]
        return extremes

    @staticmethod
    def _exclude_zeros(
        exclude: list[str | int],
        sample_data: pd.DataFrame,
        x_var: str,
        y_var: str,
        id_nr: str,
    ) -> list[str | int]:
        """Check for observations with x=0 and move to exclude list. Check that all have y_var as 0."""
        mask0 = sample_data[x_var] == 0
        zeroysum = sample_data.loc[mask0, y_var].sum()
        assert (
            zeroysum == 0
        ), f"There are observations in your sample where {x_var} is zero but {y_var} is > 0. This is not allowed in a rate model. Please adjust or remove them."

        # Add to exclude list if doesn't fail
        if mask0.sum() > 0:
            print(
                f'There are {mask0.sum()} observations in the sample with {x_var} = 0. These are moved to "surprise strata".'
            )

            exclude = exclude + sample_data.loc[mask0, id_nr].tolist()

        return exclude
