# # Code for rate estimation
# #### To do:
#
# - Fix for ulike strata i utvalg og pop
# - Adjust rerunning for excludes to only rerun specific strata.
# - Add in option for several y values.
# - Homogen model option
# - Regression model option

# +


# Import libraries
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf  # type: ignore

from .ssbmodel import ssbmodel

# -


class ratemodel(ssbmodel):
    """Class for estimating statistics for business surveys using a rate model."""

    def __init__(
        self,
        pop_data: pd.DataFrame,
        sample_data: pd.DataFrame,
        id_nr: str,
        verbose: int = 1,
    ) -> None:
        """Initialization of ratemodel object."""
        super().__init__(pop_data, sample_data, id_nr, verbose)

    def fit(
        self,
        y_var: str,
        x_var: str,
        strata_var: str | list[str] = "",
        control_extremes: bool = True,
        exclude: list[str | int] | None = None,
        exclude_auto: int = 0,
        remove_missing: bool = True,
        rbound: float = 2,
        gbound: float = 2,
        count: int = 0,
    ) -> None:
        """Run and fit a rate model within strata.

        Args:
            y_var: The target variable to estimate from the survey.
            x_var: The variable to use as the explanatory variable in the model.
            strata_var: The stratification variable.
            control_extremes: Whether the model should be fitted in a way that allows for extremes value controls.
            exclude: List of ID numbers for observations to exclude.
            exclude_auto: Whether extreme values should be automatically excluded from the models. Default 0. Integer 1 indicates extreme values should be removed once and model run again.
            remove_missing: Whether to automatically remove units in the sample that are missing x or y values.
            rbound: Multiplicative value to determine the extremity of the studentized residual values.
            gbound: Multiplicative value to determine the extremity of the G values.
            count: Integer value for the round count if using automatic exclusion of outliers.

        """
        # Check variables
        self._check_variable(x_var, self.pop_data, data_name="population")
        self._check_variable(x_var, self.sample_data, remove_missing=remove_missing)
        self._check_variable(y_var, self.sample_data, remove_missing=remove_missing)

        # Bytte dtype hvis INt64 for x siden det fungere ikke med modellen - quick fix
        # To DO: swap to funksjon
        if self.pop_data[x_var].dtype == "Int64":
            self.pop_data[x_var] = self.pop_data[x_var].astype("int64")
        if self.sample_data[x_var].dtype == "Int64":
            self.sample_data[x_var] = self.sample_data[x_var].astype("int64")
        if self.sample_data[y_var].dtype == "Int64":
            self.sample_data[y_var] = self.sample_data[y_var].astype("int64")

        # Check control and exclude_auto
        if (not control_extremes) & (exclude_auto > 0):
            print(
                "You have chosen to automatically remove outliers so extreme values must be controlled. Parameter control_extremes set to True."
            )
            control_extremes = True

        if (self.verbose == 2) & (exclude_auto > 0):
            print(f"Fitting rate model number: {exclude_auto}")

        if (self.verbose == 2) & (exclude_auto == 0):
            print("Fitting final rate model")

        # Define the model formula
        formula = y_var + "~" + x_var + "- 1"
        self.y_var = y_var
        self.x_var = x_var

        # Create stratum variables
        if not strata_var:
            self.sample_data["_stratum"] = "1"
            self.pop_data["_stratum"] = "1"
            strata_var_new: str = "_stratum"
            self.strata_var = "_stratum"
        elif isinstance(strata_var, list):
            if len(strata_var) == 1:
                strata_var_new = strata_var[0]
            else:
                self.sample_data["_stratum"] = self._fold_dataframe(
                    df=self.sample_data[strata_var]
                )
                self.pop_data["_stratum"] = self._fold_dataframe(
                    df=self.pop_data[strata_var]
                )
                strata_var_new = "_stratum"
        else:
            strata_var_new = strata_var
        self._check_variable(
            strata_var_new, self.pop_data, data_name="population", check_for_char=True
        )
        self.strata_var = strata_var_new

        # Check all strata in sample are in population
        unique_strata_pop = self.pop_data[strata_var_new].unique()
        unique_strata_sample = self.sample_data[strata_var_new].unique()

        all_values_present = all(
            value in unique_strata_sample for value in unique_strata_pop
        )
        assert (
            all_values_present
        ), "Not all strata in the population were found in the sample data. Please check."
        all_values_present = all(
            value in unique_strata_pop for value in unique_strata_sample
        )
        assert (
            all_values_present
        ), "Not all strata in the sample were found in the population data. Please check."

        # Create new strata variable for modelling in case there are excluded observations
        self.pop_data["_strata_var_mod"] = self.pop_data[strata_var_new]
        self.sample_data["_strata_var_mod"] = self.sample_data[strata_var_new]

        # Check strata is same in sample and pop for excluded units and change if necessary
        if exclude is not None:
            sample_data_filtered = self.sample_data[
                self.sample_data[self.id_nr].isin(exclude)
            ]
            pop_data_filtered = self.pop_data[self.pop_data[self.id_nr].isin(exclude)]
            merged_df = pd.merge(
                pop_data_filtered[[self.id_nr, "_strata_var_mod"]],
                sample_data_filtered[[self.id_nr, "_strata_var_mod"]],
                on=self.id_nr,
                suffixes=("_pop", "_sample"),
            )
            update_needed = (
                merged_df["_strata_var_mod_sample"] != merged_df["_strata_var_mod_pop"]
            )
            if update_needed.sum() > 0:
                print(
                    f"Stratum different in sample and population for excluded unit(s): {merged_df.loc[update_needed,self.id_nr].values}. The sample data strata will be used."
                )
                ids_to_update = merged_df.loc[update_needed, self.id_nr]
                for i in ids_to_update:
                    mask1 = self.pop_data[self.id_nr] == i
                    mask2 = merged_df[self.id_nr] == i
                    self.pop_data.loc[mask1, "_strata_var_mod"] = merged_df.loc[
                        mask2, "_strata_var_mod_sample"
                    ].values

            # Next update for strata_var_mod to surprise strata
            self.pop_data = self._update_strata(self.pop_data, exclude)
            self.sample_data = self._update_strata(self.sample_data, exclude)

        # Set up coefficient dictionaries
        strata_results: dict["str", Any] = {}  # Each stratum as key
        obs_data: dict["str", Any] = {}  # Each stratum as key

        # Iterate over each stratum in sample and fit model
        for stratum, group in self.sample_data.groupby("_strata_var_mod"):
            stratum_info: dict["str", Any] = {
                "_strata_var_mod": stratum,
                "n": len(group),  # Number of observations in the sample
                "x_sum_sample": group[x_var].sum(),
            }
            obs_info: dict["str", Any] = {
                "_strata_var_mod": stratum,
                self.id_nr: group[self.id_nr].values,
                "xvar": group[x_var],
                "yvar": group[y_var],
            }
            if len(group) > 1:  # Ensure there is more than one row to fit a model
                if self.verbose == 2:
                    print(f"\nFitting model for Stratum: {stratum!r}")

                # Adjusting weights for zero values in x variable
                weights = 1.0 / (group[x_var])
                weights.loc[group[x_var] == 0] = 1.0 / (group[x_var] + 1)

                # Fit the weighted least squares model
                model = smf.wls(formula, data=group, weights=weights).fit()
                sigma2 = (1.0 / (len(group) - 1)) * sum(
                    model.resid.values**2 / group[x_var]
                )

                # Add into stratum information
                stratum_info.update(
                    {
                        "beta": model.params[x_var].item(),  # Series of coefficients
                        "sigma2": sigma2,
                    }
                )

                # Add in residuals and hat values to observation info
                hats = self._get_hat(group[[x_var]].values, weights)
                obs_info.update({"resids": model.resid.values, "hat": hats})

                # Check for x=0 and return message
                if (any(group[x_var] == 0)) & (control_extremes):
                    print(
                        f"Values of zero detected in stratum {stratum!r}. These observations will not be assessed for extremeness."
                    )

                # Check y for all 0's and return message
                if (all(group[y_var] == 0)) & (control_extremes):
                    print(
                        f"All values for {y_var!r} in stratum {stratum!r} were zero. Extreme values need to be checked in other ways for this stratum."
                    )

                # Check y for all 0's but one
                if sum(group[y_var] == 0) == (len(group) - 1):
                    print(
                        f"Only one non-zero value found for {y_var!r} in stratum {stratum!r}. Extreme detection can't be performed for the non-zero observation."
                    )

                # Add in studentized residuals and G values if specified
                if control_extremes:
                    if len(group) == 2:
                        print(
                            f"Extreme values not able to be detected in stratum: {stratum!r} due to too few observations."
                        )
                        obs_info.update(
                            {"rstud": np.nan, "G": np.nan, "beta_ex": np.nan}
                        )
                    else:
                        rstuds = self._get_rstud(
                            y=np.array(group[y_var]),
                            res=model.resid.values,
                            x_var=x_var,
                            df=group,
                            hh=hats,
                            X=np.array(group[x_var]),
                            formula=formula,
                        )
                        obs_info.update(
                            {
                                "rstud": rstuds[0],
                                "G": rstuds[0] * (np.sqrt(hats / (1.0 - hats))),
                                "beta_ex": rstuds[1],
                            }
                        )

            else:
                if "surprise" not in stratum:  # type: ignore
                    print(
                        f"Stratum: {stratum!r}, has only one observation and has 0 variance. Consider combing strata."
                    )
                # Add standard info in for 1 obs strata : check for x-values = 0
                x = group[x_var].values[0]

                if x == 0:
                    print(
                        f"Stratum {stratum!r}, has 1 observation and has a x-value of 0. This is causing a problem with estimates and has been adjusted to 0.01."
                    )
                    x = 0.01  ## Not quite right but quick fix for errors. Need to fix properly
                stratum_info.update(
                    {
                        "beta": group[y_var].values[0] / x,
                        "sigma2": 0,
                    }
                )
                obs_info.update({"resids": [0], "hat": np.nan})
                if control_extremes:
                    obs_info.update({"rstud": np.nan, "G": np.nan, "beta_ex": np.nan})
            strata_results[stratum] = stratum_info  # type: ignore
            obs_data[stratum] = obs_info  # type: ignore

        if self.verbose == 2:
            print("Finished fitting models. Now summarizing additional data")

        # Loop through population also
        for stratum, group in self.pop_data.groupby("_strata_var_mod"):
            stratum_info = {"N": len(group), "x_sum_pop": group[x_var].sum()}
            # Condition to see if strata exists. This is for cases where excludes observations are already excluded due to missing data
            if stratum in strata_results:
                strata_results[stratum].update(stratum_info)  # type: ignore

        # Set results to instance
        self.strata_results = strata_results
        self.obs_data = obs_data

        # Check for outliers and re-run if auto exclude is on
        if exclude_auto > 0:
            count = +1
            extremes = self.get_extremes(rbound=rbound, gbound=gbound)[
                self.id_nr
            ].values.tolist()
            print(f"The following were extreme values and were excluded: {extremes!r}")
            if exclude is None:
                exclude = extremes
            else:
                exclude = exclude + extremes
            exclude_auto -= 1

            self.fit(
                y_var=y_var,
                x_var=x_var,
                strata_var=strata_var,
                control_extremes=control_extremes,
                exclude=exclude,
                exclude_auto=exclude_auto,
                remove_missing=remove_missing,
                rbound=rbound,
                gbound=gbound,
                count=count,
            )

    @staticmethod
    def _fold_dataframe(df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        """This function folds all Series in a DataFrame into one Series with concatenated strings.

        For every series in the df, it will convert the rows to strings, and then repeatedly
        concatenate the strings for every row in every series.

        """
        series: list[pd.Series] = [df[col] for col in df]  # type: ignore[type-arg]
        concat_series = series[0].astype(str).str.cat(others=series[1:], sep="_")
        return concat_series

    def _update_strata(
        self, df: pd.DataFrame, exclude: list[str | int]
    ) -> pd.DataFrame:
        """Update files to include a new variable for modelling including suprise strata."""
        # Use the 'loc' method to locate the rows where ID is in the exclude list and update 'strata'
        mask = df[self.id_nr].isin(exclude)
        df.loc[mask, "_strata_var_mod"] = (
            df.loc[mask, "_strata_var_mod"]
            + "_surprise_"
            + df.loc[mask, self.id_nr].astype(str)
        )
        return df

    def _get_hat(self, X: Any, W: Any) -> Any:
        """Get the hat matrix for the model."""
        # Compute the square root of the weight matrix, W^(1/2)
        W = np.diag(W)
        W_sqrt = np.sqrt(W)

        # Compute (X^T * W * X)^(-1)
        XTWX_inv = np.linalg.inv(X.T @ W @ X)

        # Compute the hat matrix
        H = W_sqrt @ X @ XTWX_inv @ X.T @ W_sqrt

        # Return diagonal
        return np.diag(H)

    def _get_rstud(
        self,
        y: Any,
        res: Any,
        x_var: str,
        df: pd.DataFrame,
        hh: Any,
        X: Any,
        formula: str,
    ) -> tuple[Any, Any]:
        """Get the studentized residuals from the model."""
        # set up vectors
        n = len(y)
        y = np.array(y)
        X = np.array(X)
        beta_ex_values = np.zeros(n)
        R = np.zeros(n)

        for i in range(n):
            # Exclude the i-th observation
            df_i = df.drop(index=df.iloc[i].name)
            ww_i = 1.0 / (df_i[x_var])
            ww_i.loc[df_i[x_var] == 0] = 1.0 / (df_i[x_var] + 1)
            X_i = np.delete(X, i, axis=0)
            y_i = np.delete(y, i, axis=0)

            # Fit the WLS model without the i-th observation
            model_i = smf.wls(formula, data=df_i, weights=ww_i).fit()
            beta_ex_values[i] = model_i.params[x_var].item()

            # get predicted values y_j
            y_hat_j = model_i.predict(df_i)

            # Calculate sigma
            sigma2 = sum((y_i - y_hat_j) ** 2 / X_i) * 1.0 / (n - 2)

            # Calculate and save studentized residuals
            if (X[i] > 0) & (sigma2 > 0):
                R[i] = res[i] / (np.sqrt(sigma2 * X[i]) * np.sqrt(1.0 - hh[i]))
            else:
                R[i] = np.nan

        return (R, beta_ex_values)

    @property
    def get_coeffs(self) -> pd.DataFrame:
        """Get the model coefficients for each strata."""
        return pd.DataFrame(self.strata_results).T

    @property
    def get_obs(self) -> dict[str, Any]:
        """Get the details for observations from the model."""
        return self.obs_data

    def _get_robust(self, strata: str) -> tuple[float, float, float]:
        """Get robust variance estimations."""
        # collect data for strata
        x_pop = self.strata_results[strata]["x_sum_pop"]
        x_utv = self.strata_results[strata]["x_sum_sample"]

        hi = self.obs_data[strata]["hat"]
        ei = self.obs_data[strata]["resids"]

        if (isinstance(ei, (pd.Series | np.ndarray))) & (
            isinstance(hi, (pd.Series | np.ndarray))
        ):
            # Calculate ai
            Xr = x_pop - x_utv
            ai = Xr / x_utv

            # Calculate di variations
            di_1 = ei**2
            di_2 = ei**2 / (1.0 - hi)
            di_3 = ei**2 / ((1.0 - hi) ** 2)

            # Caluclate variances
            V1 = sum(ai**2 * di_1) + sum(di_1) * ai
            V2 = sum(ai**2 * di_2) + sum(di_2) * ai
            V3 = sum(ai**2 * di_3) + sum(di_3) * ai

            # Adjust negative variance to zero
            if any(x < 0 for x in [V1, V2, V3]):
                if self.verbose == 2:
                    print(
                        "Negative variances calculated. These are being adjusted to 0."
                    )
                V1 = max(V1, 0)
                V2 = max(V2, 0)
                V3 = max(V3, 0)

            return (V1, V2, V3)
        else:
            return (0, 0, 0)

    def _get_standard_variance(self, strata: str) -> Any:
        """Get standard variance estimates."""
        x_pop = self.strata_results[strata]["x_sum_pop"]
        x_utv = self.strata_results[strata]["x_sum_sample"]
        s2 = self.strata_results[strata]["sigma2"]

        if (x_pop > 0) and (x_utv > 0):
            V = x_pop**2 * (x_pop - x_utv) / x_pop * s2 / x_utv
        else:
            V = np.nan

        if V < 0:
            if self.verbose == 2:
                print("Negative variances calculated. These are being adjusted to 0.")
            V = 0

        return V

    def _get_domain_estimates(self, domain: str, uncertainty_type: str) -> pd.DataFrame:
        """Get domain estimation for case where domains are not an aggregation of strata."""
        # Collect data
        self._add_flag()
        pop = self.get_imputed()  # add imputed y values
        res = self.strata_results  # get results for sigma2 values
        strata_var = self.strata_var  # use variable without surprise strata

        # Create domain and strata lists
        domain_unique = pop[domain].unique().tolist()
        strata_unique = pop[strata_var].unique().tolist()
        domain_df = {}

        # loop through domains and calculate variance
        for d in domain_unique:
            temp_dom = pop.loc[pop[domain] == d]

            # Get additional domain information on sample, pop sizes and estimates
            N = temp_dom.shape[0]
            n = np.sum(temp_dom[self.flag_var] == 1)
            est = np.sum(temp_dom[f"{self.y_var}_imp"])

            # Loop through strata to get the partial variances
            var = 0
            x_sum_sample = 0
            x_sum_pop = 0
            for s in strata_unique:
                mask_s = (temp_dom[strata_var] == s) & (
                    temp_dom[self.flag_var] == 0
                )  # Those not in sample
                Uh_sh = np.sum(
                    temp_dom.loc[mask_s, self.x_var]
                )  # Sum of x not in sample
                xh = res[s]["x_sum_sample"]  # Sum of x in sample
                s2 = res[s]["sigma2"]
                x_sum_pop += np.sum(temp_dom.loc[temp_dom[strata_var] == s, self.x_var])
                x_sum_sample += np.sum(
                    temp_dom.loc[
                        (temp_dom[strata_var] == s) & (temp_dom[self.flag_var] == 1),
                        self.x_var,
                    ]
                )

                # Add in variance for stratum if sum of x is greater than 0 and var is not na or inf!!!
                if (xh > 0) & (not np.isinf(s2)) & (not np.isnan(s2)):
                    var += s2 * Uh_sh * ((Uh_sh + xh) / xh)

            # Add calculations to domain dict
            domain_df[d] = {
                "domain": d,
                "N": N,
                "n": n,
                f"{self.x_var}_sum_pop": x_sum_pop,
                f"{self.x_var}_sum_sample": x_sum_sample,
                f"{self.y_var}_EST": est,
                f"{self.y_var}_VAR": var,
            }

        # Convert to pandas
        domain_pd = pd.DataFrame([v for k, v in domain_df.items()])
        domain_pd = self._clean_output(
            domain_pd,
            uncertainty_type=uncertainty_type,
            variance_type="standard",
            return_type="unbiased",
        )

        return domain_pd

    def _clean_output(
        self,
        result: pd.DataFrame,
        uncertainty_type: str,
        variance_type: str,
        return_type: str,
    ) -> pd.DataFrame:
        """Clean up results set to include the chosen return type."""
        y_var = self.y_var

        # Format and add in CV, SE, CI
        if variance_type == "standard":
            variance_list = [""]
        if (variance_type == "robust") & (return_type == "unbiased"):
            variance_list = ["2"]
        if (variance_type == "robust") & (return_type == "all"):
            variance_list = ["1", "2", "3"]

        for i in variance_list:
            if "CV" in uncertainty_type:
                result[f"{y_var}_CV{i}"] = (
                    np.sqrt(result[f"{y_var}_VAR{i}"]) / result[f"{y_var}_EST"] * 100
                )

            if "SE" in uncertainty_type:
                result[f"{y_var}_SE{i}"] = np.sqrt(result[f"{y_var}_VAR{i}"])

            if "CI" in uncertainty_type:
                result[f"{y_var}_LB{i}"] = result[f"{y_var}_EST"] - (
                    1.96 * np.sqrt(result[f"{y_var}_VAR{i}"])
                )
                result[f"{y_var}_UB{i}"] = result[f"{y_var}_EST"] + (
                    1.96 * np.sqrt(result[f"{y_var}_VAR{i}"])
                )

            if "VAR" not in uncertainty_type:
                result = result.drop([f"{y_var}_VAR{i}"], axis=1)

            if (return_type == "unbiased") & (variance_type == "robust"):
                result = result.drop([f"{y_var}_VAR1"], axis=1)
                result = result.drop([f"{y_var}_VAR3"], axis=1)

        return result

    def _get_domain(self, domain: str) -> Any:
        """Get mapping of domain to the strata results."""
        strata_var = "_strata_var_mod"

        # create key form population file
        pop = self.pop_data[[strata_var, domain]]
        aggregated = pop.groupby(strata_var)[domain].agg(list)

        domain_key = aggregated.to_dict()
        for strata_var, domain in domain_key.items():
            domain_key[strata_var] = list(set(domain))

        # check for 1 to many and return error
        one_to_many = {
            strata_var: domains
            for strata_var, domains in domain_key.items()
            if len(set(domains)) > 1
        }
        assert (
            len(one_to_many) == 0
        ), "1-to-many relationship(s) found in strata/domain aggregation. Only aggregated variances are programmed."

        # map key
        strata_res = pd.DataFrame(self.strata_results).T
        domain_mapped = strata_res["_strata_var_mod"].map(domain_key)

        return domain_mapped.str[0]

    def get_estimates(
        self,
        domain: str = "",
        uncertainty_type: str = "CV",
        variance_type: str = "robust",
        return_type: str = "unbiased",
    ) -> pd.DataFrame:
        """Get estimates for previously run model within strata or domains. Variance and CV estimates are returned for each domain.

        Args:
            domain: Name of the variable to use for estimation. Should be in the population data.
            uncertainty_type: Which uncertainty measures to return. Choose between 'CV' (default) for coefficient of variation, 'VAR' for variance, 'SE' for standard errors, 'CI' for confidence intervals. Multiple measures can be returned with combinations of these, for example "CV_SE" returns both the coefficient of variation and the standard error.
            variance_type: Choose from 'robust' or 'standard' estimation of variance. Currently only robust estimation is calculated for strata and aggregated strata domains estimation and standard for other domains.
            return_type: String for which robust estimates to return. Choose 'unbiased' to return only the unbiased robust variance estimate or 'all' to return all three.

        Returns:
            A pd.Dataframe is returned conatining estimates and variance/coefficient of variation estimations for each domain.

        """
        # Check model is run
        self._check_model_run()

        # Fetch results
        strata_df = pd.DataFrame(self.strata_results).T

        # Add in domain
        if not domain:
            domain = self.strata_var
        try:
            strata_df[domain] = self._get_domain(domain)

        # If domains are not aggregates of strata run alternative calculation for variance (not robust)
        except AssertionError:
            if variance_type == "robust":
                print(
                    "Domain variable is not an aggregation of strata variables. Only standard variance calculations are available."
                )
            return self._get_domain_estimates(domain, uncertainty_type)

        # Format variables
        strata_df["N"] = pd.to_numeric(strata_df["N"])
        strata_df["n"] = pd.to_numeric(strata_df["n"])

        strata_df[f"{self.x_var}_sum_pop"] = pd.to_numeric(strata_df["x_sum_pop"])
        strata_df[f"{self.x_var}_sum_sample"] = pd.to_numeric(strata_df["x_sum_sample"])

        # Add estimates
        strata_df["beta"] = pd.to_numeric(strata_df["beta"])
        strata_df[f"{self.y_var}_EST"] = pd.to_numeric(
            strata_df["beta"] * strata_df["x_sum_pop"]
        )

        # Add variance standard or robust
        if variance_type == "standard":
            var1 = []
            for s in strata_df["_strata_var_mod"]:
                var1.append(self._get_standard_variance(s))

            strata_df[f"{self.y_var}_VAR"] = np.array(var1)

            # Aggregate to domain
            result = (
                strata_df[[domain, "N", "n", f"{self.y_var}_EST", f"{self.y_var}_VAR"]]
                .groupby(domain)
                .sum()
            )
        if variance_type == "robust":
            var1 = []
            var2 = []
            var3 = []
            for s in strata_df["_strata_var_mod"]:
                var = self._get_robust(s)
                if isinstance(var, tuple):
                    var1.append(var[0])
                    var2.append(var[1])
                    var3.append(var[2])

            # Add to results
            variables = ["VAR1", "VAR2", "VAR3"]
            variance_list = [var1, var2, var3]
            for var_name, data in zip(variables, variance_list, strict=False):
                strata_df[f"{self.y_var}_{var_name}"] = data

            # Aggregate to domain
            result = (
                strata_df[
                    [
                        domain,
                        "N",
                        "n",
                        f"{self.x_var}_sum_pop",
                        f"{self.x_var}_sum_sample",
                        f"{self.y_var}_EST",
                        f"{self.y_var}_VAR1",
                        f"{self.y_var}_VAR2",
                        f"{self.y_var}_VAR3",
                    ]
                ]
                .groupby(domain)
                .sum()
            )

        # Format and add in CV, SE, CI
        result = self._clean_output(
            result,
            uncertainty_type=uncertainty_type,
            variance_type=variance_type,
            return_type=return_type,
        )

        return result

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
            new["beta"] = self.strata_results[k]["beta"]
            new["_strata_var_mod"] = self.strata_results[k]["_strata_var_mod"]
            new["n"] = self.strata_results[k]["n"]
            new["N"] = self.strata_results[k]["N"]
            new[self.x_var] = new["xvar"]
            new[self.y_var] = new["yvar"]
            new["x_sum_pop"] = self.strata_results[k]["x_sum_pop"]
            new[f"{self.y_var}_EST"] = new["beta"] * new["x_sum_pop"]
            new[f"{self.y_var}_EST_ex"] = new["beta_ex"] * new["x_sum_pop"]
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

    def get_imputed(self) -> pd.DataFrame:
        """Get population data with imputed values based on model.

        Returns:
            Pandas data frame with all in population and imputed values
        """
        self._check_model_run()

        self._add_flag()
        pop = self.pop_data
        utvalg = self.sample_data

        # Map beta values to the population file
        def _get_beta(stratum: str) -> Any:
            """Get beta values from model for stratum."""
            return self.strata_results.get(stratum, {}).get("beta", None)

        pop["beta"] = pop["_strata_var_mod"].apply(_get_beta)

        # Calculate imputed values
        pop[f"{self.y_var}_imp"] = pop["beta"] * pop[self.x_var]

        # Link in survey values
        id_to_yvar_map = utvalg.set_index(self.id_nr)[self.y_var]
        pop[f"{self.y_var}_imp"] = (
            pop[self.id_nr].map(id_to_yvar_map).fillna(pop[f"{self.y_var}_imp"])
        )
        pop_pd = pd.DataFrame(pop)
        return pop_pd

    def get_weights(self) -> pd.DataFrame:
        """Get sample data with weights based on model.

        Returns:
            Pandas data frame with sample data and weights.
        """
        self._check_model_run()

        utvalg = self.sample_data

        # map population and sample x totals to survey data
        def _get_sums(stratum: str, var: str) -> Any:
            """Get sums within strata."""
            sum_value = self.strata_results.get(stratum, {}).get(var, None)
            return sum_value

        # Apply the function to create a new 'beta' column in the DataFrame. Use strata_var_mod to consider surprise strata
        sample_sum = utvalg["_strata_var_mod"].apply(_get_sums, var="x_sum_sample")
        pop_sum = utvalg["_strata_var_mod"].apply(_get_sums, var="x_sum_pop")

        utvalg["estimation_weights"] = pop_sum / sample_sum

        # Check for obs in suprise strata and set to 1
        mask = utvalg._strata_var_mod.str.contains("surprise")
        utvalg.loc[mask, "estimation_weights"] = 1

        return utvalg

    def _check_extreme_run(self) -> None:
        """Check to ensure that extreme value requirements were run during fitting."""
        self._check_model_run()

        is_rstud_present = "rstud" in next(iter(self.obs_data.values()))

        if not is_rstud_present:
            raise RuntimeError(
                "Model has not been fitted for calculating extreme values. Please re-run fit() with control_extremes = True"
            )

    def _add_flag(self) -> None:
        """Add flag in population data to say if unit is in the sample or not."""
        self.flag_var = f"{self.y_var}_flag_sample"
        sample_ids = set(self.sample_data[self.id_nr])
        self.pop_data[self.flag_var] = self.pop_data[self.id_nr].apply(
            lambda x: 1 if x in sample_ids else 0
        )
