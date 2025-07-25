# # Code for estimation using a regression model


# Import libraries
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf  # type: ignore

from .stratifiedmodel import StratifiedModel


# Main class for Ratio model
class RegModel(StratifiedModel):
    """Class for estimating statistics for business surveys using a ratio model."""

    def fit(
        self,
        model_formula: str,
        strata_var: str | list[str] = "",
        control_extremes: bool = True,
        exclude: list[str | int] | None = None,
        remove_missing: bool = True,
    ) -> None:
        """Run and fit a ratio model within strata.

        Args:
            model_formula: The formula to use for the regression model.
            strata_var: The stratification variable.
            control_extremes: Whether the model should be fitted in a way that allows for extremes value controls.
            exclude: List of ID numbers for observations to exclude.
            remove_missing: Whether to automatically remove units in the sample that are missing x or y values.
        """
        # Extract x and y variables for checking 
        y_var, x_var = self.extract_vars(model_formula)

        # Check variables
        for x in x_var:
            self._check_variable(x, self.pop_data, data_name="population")
            self._check_variable(x, self.sample_data, remove_missing=remove_missing)

            # Convert type if necessary
            self.pop_data[x] = self._convert_var(x, self.pop_data)
            self.sample_data[x] = self._convert_var(x, self.sample_data)
        self._check_variable(y_var, self.sample_data, remove_missing=remove_missing)
        self.sample_data[y_var] = self._convert_var(y_var, self.sample_data)



#### old below - needs changing ####
        # Fit model
        super()._fit(
            y_var=y_var,
            x_var=x_var,
            strata_var=strata_var,
            control_extremes=control_extremes,
            exclude=exclude,
            remove_missing=remove_missing,
            method="ratio",
            method_function=self._ratio_method,
            ai_function=self._get_ai_ratio,
        )
    
    @staticmethod
    def extract_vars(model_formula):
        lhs, rhs = form.split('~')
        x_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', rhs))
        return(lhs, x_vars)

    def _ratio_method(
        self,
        stratum: str,
        group: pd.DataFrame,
        stratum_info: dict[str, Any],
        obs_info: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Ratio model method."""
        formula = self.y_var + "~" + self.x_var + "- 1"

        # Create weights
        weights = 1.0 / (group[self.x_var])

        # Adjust weights if x=0
        zero_cond = group[self.x_var] == 0
        if zero_cond.sum() > 0:
            weights.loc[zero_cond] = 1.0 / (group[self.x_var] + 1)
            mes_zero = f"Zero counts were recorded in the x variable in stratum {stratum!r} and were adjusted to 1 for calculations. Extreme controls will not be done for these observations."
            self.logger.info(mes_zero)

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
                self.logger.info(
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
                    x=np.array(group[self.x_var]),
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

    def _get_ai_ratio(self, strata: str) -> Any:
        """Get ai values for robust variance for ratio model."""
        # collect data for strata
        x_pop = self.strata_results[strata][f"{self.x_var}_sum_pop"]
        x_utv = self.strata_results[strata][f"{self.x_var}_sum_sample"]

        # check for x_utv = 0
        if x_utv == 0:
            ai = 0
        # Calculate ai
        else:
            xr = x_pop - x_utv
            ai = xr / x_utv

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
        extremes_output: pd.DataFrame = extremes[
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
        return extremes_output


# For back compatibility
ratemodel = RatioModel
