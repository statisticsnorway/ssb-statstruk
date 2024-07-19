# General code for stratified models
# To do:
# robust variance implementation

# Import libraries
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf  # type: ignore

from .ssbmodel import ssbmodel


class stratifiedmodel(ssbmodel):
    """Class for estimating statistics for business surveys using a stratified model."""

    def __init__(
        self,
        pop_data: pd.DataFrame,
        sample_data: pd.DataFrame,
        id_nr: str,
        verbose: int = 1,
    ) -> None:
        """Initialization of object."""
        super().__init__(pop_data, sample_data, id_nr, verbose)

    def _fit(
        self,
        method_function: Any,
        y_var: str,
        x_var: str = "",
        strata_var: str | list[str] = "",
        control_extremes: bool = True,
        exclude: list[str | int] | None = None,
        remove_missing: bool = True,
        rbound: float = 2,
        gbound: float = 2,
        method: str = "rate",
    ) -> None:
        """Run and fit a rate model within strata.

        Args:
            method_function: Function to use for method.
            y_var: The target variable to estimate from the survey.
            x_var: The variable to use as the explanatory variable in the model.
            strata_var: The stratification variable.
            control_extremes: Whether the model should be fitted in a way that allows for extremes value controls.
            exclude: List of ID numbers for observations to exclude.
            remove_missing: Whether to automatically remove units in the sample that are missing x or y values.
            rbound: Multiplicative value to determine the extremity of the studentized residual values.
            gbound: Multiplicative value to determine the extremity of the G values.
            method: model method to use
        """
        # Check variables
        self._check_variable(y_var, self.sample_data, remove_missing=remove_missing)

        # Swap dtype if Int64 or Float64 to int64 or float64 (since these don't work with some of the models
        self.sample_data[y_var] = self._convert_var(y_var, self.sample_data)

        # Set variables
        self.y_var = y_var
        self.x_var = x_var
        self.control_extremes = control_extremes
        self.method = method

        # Create strata variable for modelling including 'surprise strata'
        strata_var_new = self._create_strata(strata_var)

        # Check strata for that they are all represented in sample and population file.
        # Also check unit level for strata differences in excludes list (not all).
        self._check_strata(strata_var_new, exclude)

        # Update for strata_var_mod to surprise strata for those in exclude list
        self.pop_data = self._update_strata(self.pop_data, exclude)
        self.sample_data = self._update_strata(self.sample_data, exclude)

        # Set up coefficient dictionaries
        strata_results: dict[str, Any] = {}  # Each stratum as key
        obs_data: dict[str, Any] = {}  # Each stratum as key
        one_nonzero_strata: list[Any] = []  # ratemodel

        # Iterate over each stratum in sample and fit model
        for stratum, group in self.sample_data.groupby("_strata_var_mod"):
            stratum_info, obs_info, one_nonzero = self._fit_model_and_controls(
                stratum, group, method_function
            )

            strata_results[stratum] = stratum_info  # type: ignore
            obs_data[stratum] = obs_info  # type: ignore

            if one_nonzero:
                one_nonzero_strata.append(stratum)

        # print check for one nonzero
        if (len(one_nonzero_strata) > 0) & (method == "rate") & (control_extremes):
            print(
                f"Only one non-zero value found for {self.y_var!r} in strata: {one_nonzero_strata}. Extreme detection can't be performed for the non-zero observations."
            )

        if self.verbose == 2:
            print("Finished fitting models. Now summarizing additional data")

        # Loop through population also to get sums
        for stratum, group in self.pop_data.groupby("_strata_var_mod"):
            stratum_info = {"N": len(group)}

            if self.x_var:
                stratum_info.update({f"{self.x_var}_sum_pop": group[x_var].sum()})

            # Condition to see if strata exists. This is for cases where excludes observations are already excluded due to missing data
            if stratum in strata_results:
                strata_results[stratum].update(stratum_info)  # type: ignore

        # Set results to instance
        self.strata_results = strata_results
        self.obs_data = obs_data

    @property
    def get_coeffs(self) -> pd.DataFrame:
        """Get the model coefficients for each strata."""
        return pd.DataFrame(self.strata_results).T

    @property
    def get_obs(self) -> dict[str, Any]:
        """Get the details for observations from the model."""
        return self.obs_data

        # Create stratum variables

    def _create_strata(self, strata_var: str | list[str]) -> str:
        """Function to create a strata variable and fix if missing or a list."""
        # If strata is missing ("") then set up a strataum variable ='1' for all
        if not strata_var:
            self.sample_data["_stratum"] = "1"
            self.pop_data["_stratum"] = "1"

            strata_var_new: str = "_stratum"
            self.strata_var = "_stratum"

        # If strata is a list then paste the variables together as one variable
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

        # Create new strata variable for modelling in case there are excluded observations
        self.strata_var = strata_var_new
        self.pop_data.loc[:, "_strata_var_mod"] = self.pop_data[strata_var_new].copy()
        self.sample_data.loc[:, "_strata_var_mod"] = self.sample_data[
            strata_var_new
        ].copy()

        return strata_var_new

    def _check_strata(
        self, strata_var_new: str, exclude: list[str | int] | None
    ) -> None:
        """Check strata variable for validity and that all found in sample and pop. Update units that are different if they are own strata."""
        self._check_variable(
            strata_var_new, self.pop_data, data_name="population", check_for_char=True
        )

        # Check all strata in sample are in population and vice-versa
        unique_strata_pop = self.pop_data[strata_var_new].unique()
        unique_strata_sample = self.sample_data[strata_var_new].unique()

        all_values_present_sample = all(
            value in unique_strata_sample for value in unique_strata_pop
        )
        all_values_present_pop = all(
            value in unique_strata_pop for value in unique_strata_sample
        )
        assert (
            all_values_present_sample
        ), "Not all strata in the population were found in the sample data. Please check."
        assert (
            all_values_present_pop
        ), "Not all strata in the sample were found in the population data. Please check."

        # Check strata is same in sample and pop for excluded units and change if necessary
        if exclude:
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
        self, df: pd.DataFrame, exclude: list[str | int] | None
    ) -> pd.DataFrame:
        """Update files to include a new variable for modelling including suprise strata."""
        # Use the 'loc' method to locate the rows where ID is in the exclude list and update 'strata'
        if exclude:
            mask = df[self.id_nr].isin(exclude)
            df.loc[mask, "_strata_var_mod"] = (
                df.loc[mask, "_strata_var_mod"]
                + "_surprise_"
                + df.loc[mask, self.id_nr].astype(str)
            )
        return df

    @staticmethod
    def _get_hat(X: Any, W: Any) -> Any:
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

    @staticmethod
    def _get_rstud(
        y: Any,
        res: Any,
        x_var: str,
        df: pd.DataFrame,
        hh: Any,
        X: Any,
        formula: str,
    ) -> tuple[Any, Any]:
        """Get the external studentized residuals from the model (exact method)."""
        # set up vectors
        n = len(y)
        y = np.array(y)
        X = np.array(X)
        beta_ex_values = np.zeros(n)
        R = np.zeros(n)

        # check for x = 0 observations
        assert np.all(
            X != 0
        ), "Studentized residuals not calculated as some oberservations have x=0."

        for i in range(n):
            # Exclude the i-th observation
            df_i = df.drop(index=df.iloc[i].name)
            ww_i = 1.0 / (df_i[x_var])
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

    def _fit_model_and_controls(
        self, stratum: Any, group: pd.DataFrame, method_function: Any
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Fit model and return result output and extreme controls."""
        # Set one non-zero y as blank
        one_nonzero_strata = False

        # set up dictionaries
        stratum_info: dict[str, Any] = {
            "_strata_var_mod": stratum,
            "n": len(group),  # Number of observations in the sample
        }

        obs_info: dict[str, Any] = {
            "_strata_var_mod": stratum,
            self.id_nr: group[self.id_nr].values,
            "yvar": group[self.y_var],
        }

        if self.method == "rate":
            stratum_info.update({f"{self.x_var}_sum_sample": group[self.x_var].sum()})
            obs_info.update({"xvar": group[self.x_var]})

        if len(group) > 1:  # Ensure there is more than one row to fit a model
            if self.verbose == 2:
                print(f"Fitting model for Stratum: {stratum!r}")

            stratum_info, obs_info = method_function(
                stratum, group, stratum_info, obs_info
            )

            # if self.method == "rate":
            #    stratum_info, obs_info = self._rate_method(
            #        stratum, group, stratum_info, obs_info
            #    )

            # elif self.method == "homogen":
            #    stratum_info, obs_info = self._homogen_method(
            #        stratum, group, stratum_info, obs_info
            #    )

            # Check y for all 0's and return message
            if (all(group[self.y_var] == 0)) & (self.control_extremes):
                print(
                    f"All values for {self.y_var!r} in stratum {stratum!r} were zero. Extreme values need to be checked in other ways for this stratum."
                )

            # Check y for all 0's but one
            if sum(group[self.y_var] == 0) == (len(group) - 1):
                one_nonzero_strata = True

        else:
            if "surprise" not in stratum:
                print(
                    f"Stratum: {stratum!r}, has only one observation and has 0 variance. Consider combing strata."
                )
            if self.verbose == 2:
                print(f"Adding in 1 observation stratum: {stratum!r}")

            # Set x=1 so doesn't produce error. beta will still be 0 for x=0 as y must = 0 for rate model - !!need to check for
            if not self.x_var:
                x = 1
            elif group[self.x_var].values[0] == 0:
                x = 1
            else:
                # Add standard info in for 1 obs strata : check for x-values = 0
                x = group[self.x_var].values[0]

            stratum_info.update(
                {
                    f"{self.y_var}_beta": group[self.y_var].values[0] / x,
                    "sigma2": 0,
                }
            )
            obs_info.update({"resids": [0], "hat": np.nan})
            if self.control_extremes:
                obs_info.update(
                    {"rstud": np.nan, "G": np.nan, f"{self.y_var}_beta_ex": np.nan}
                )

        return stratum_info, obs_info, one_nonzero_strata

    def _get_estimates(
        self,
        ai_function: Any,
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
            ai_function: Internal function

        Returns:
            A pd.Dataframe is returned conatining estimates and variance/coefficient of variation estimations for each domain.

        """
        # Check model is run
        self._check_model_run()

        # Check output type
        assert output_type in [
            "table",
            "weights",
            "imputed",
        ], "output_type should be 'table', 'weights', or 'imputed'"

        # Fetch results
        strata_df = pd.DataFrame(self.strata_results).T

        # Add in domain
        is_aggregate = True
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

            is_aggregate = False
            variance_type = "standard"
            strata_df = self._get_domain_estimates(domain, uncertainty_type)

        # Format variables
        strata_df["N"] = pd.to_numeric(strata_df["N"])
        strata_df["n"] = pd.to_numeric(strata_df["n"])

        if self.method == "rate":
            strata_df[f"{self.x_var}_sum_pop"] = pd.to_numeric(
                strata_df[f"{self.x_var}_sum_pop"]
            )
            strata_df[f"{self.x_var}_sum_sample"] = pd.to_numeric(
                strata_df[f"{self.x_var}_sum_sample"]
            )
            x_pop = strata_df[f"{self.x_var}_sum_pop"]
        elif self.method == "homogen":
            x_pop = strata_df["N"]

        # Add estimates if aggregate
        if is_aggregate:
            strata_df[f"{self.y_var}_beta"] = pd.to_numeric(
                strata_df[f"{self.y_var}_beta"]
            )
            # strata_df.rename(columns={"beta": f"{self.y_var}_beta"}, inplace=True)
            strata_df[f"{self.y_var}_EST"] = pd.to_numeric(
                strata_df[f"{self.y_var}_beta"] * x_pop
            )

            # Add variance
            strata_df = self._add_variance(
                strata_df, domain, variance_type, ai_function
            )

        # Format and add in CV, SE, CI
        result = self._clean_output(
            strata_df,
            uncertainty_type=uncertainty_type,
            variance_type=variance_type,
            return_type=return_type,
        )

        if output_type == "table":
            merged_file = result.copy()

        elif output_type == "weights":
            sample_file = self.get_weights()
            merged_file = sample_file.merge(
                result, left_on=domain, right_on="domain", how="left"
            )
            merged_file.drop(columns=["domain"], inplace=True)

        else:  # output_type == 'imputed'
            imputed_file = self.get_imputed()
            merged_file = imputed_file.merge(
                result, left_on=domain, right_on="domain", how="left"
            )
            merged_file.drop(columns=["domain"], inplace=True)

        return merged_file

    def _add_variance(
        self,
        strata_df: pd.DataFrame,
        domain: str,
        variance_type: str,
        ai_function: Any,
    ) -> pd.DataFrame:
        """Add standard or robust variance to table."""
        # Add variance standard or robust
        if variance_type == "standard":
            var1 = []
            for s in strata_df["_strata_var_mod"]:
                var1.append(self._get_standard_variance(s))

            strata_df[f"{self.y_var}_VAR"] = np.array(var1)

            # Aggregate to domain
            selected_columns = [
                domain,
                "N",
                "n",
                f"{self.y_var}_EST",
                f"{self.y_var}_VAR",
            ]
            if self.method == "rate":
                selected_columns = (
                    selected_columns[:3]
                    + [f"{self.x_var}_sum_pop", f"{self.x_var}_sum_sample"]
                    + selected_columns[3:]
                )
            result = strata_df[selected_columns].groupby(domain).sum()
        if variance_type == "robust":
            var1 = []
            var2 = []
            var3 = []
            for s in strata_df["_strata_var_mod"]:
                var = self._get_robust(s, ai_function)
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
            selected_columns = [
                domain,
                "N",
                "n",
                f"{self.y_var}_EST",
                f"{self.y_var}_VAR1",
                f"{self.y_var}_VAR2",
                f"{self.y_var}_VAR3",
            ]
            if self.method == "rate":
                selected_columns = (
                    selected_columns[:3]
                    + [f"{self.x_var}_sum_pop", f"{self.x_var}_sum_sample"]
                    + selected_columns[3:]
                )
            result = strata_df[selected_columns].groupby(domain).sum()

        return result

    def _get_domain_estimates(self, domain: str, uncertainty_type: str) -> pd.DataFrame:
        """Get domain estimation for case where domains are not an aggregation of strata."""
        if self.method == "homogen":
            raise AssertionError(
                "Standard variance not programmed yet for homogen model domains"
            )

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
                xh = res[s][f"{self.x_var}_sum_sample"]  # Sum of x in sample
                s2 = res[s]["sigma2"]
                x_sum_pop += np.sum(temp_dom.loc[temp_dom[strata_var] == s, self.x_var])
                x_sum_sample += np.sum(
                    temp_dom.loc[
                        (temp_dom[strata_var] == s) & (temp_dom[self.flag_var] == 1),
                        self.x_var,
                    ]
                )

                # Add in variance for stratum if var is not na or inf
                if xh == 0:
                    xh = 1
                    print(
                        f"The expanatory variable (x_var) in domain, {d}, summed to zero. This has been adjusted to 1 for variance calculations"
                    )
                if (not np.isinf(s2)) & (not np.isnan(s2)):
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

        return domain_pd

    def _get_standard_variance(self, strata: str) -> Any:
        """Get standard variance estimates."""
        s2 = self.strata_results[strata]["sigma2"]
        if self.method == "rate":
            x_pop = self.strata_results[strata][f"{self.x_var}_sum_pop"]
            x_utv = self.strata_results[strata][f"{self.x_var}_sum_sample"]

        elif self.method == "homogen":
            x_pop = self.strata_results[strata]["N"]  # Here we use counts instead of x
            x_utv = self.strata_results[strata]["n"]

        if (x_pop > 0) and (x_utv > 0):
            V = x_pop**2 * (x_pop - x_utv) / x_pop * s2 / x_utv
        else:
            V = np.nan

        if V < 0:
            if self.verbose == 2:
                print("Negative variances calculated. These are being adjusted to 0.")
            V = 0

        return V

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

    def _check_extreme_run(self) -> None:
        """Check to ensure that extreme value requirements were run during fitting."""
        self._check_model_run()

        is_rstud_present = "rstud" in next(iter(self.obs_data.values()))

        if not is_rstud_present:
            raise RuntimeError(
                "Model has not been fitted for calculating extreme values. Please re-run fit() with control_extremes = True"
            )

    def _get_robust(
        self, strata: str, ai_function: Any
    ) -> tuple[float, float, float]:
        """Get robust variance estimations."""
        hi = self.obs_data[strata]["hat"]
        ei = self.obs_data[strata]["resids"]
        ai = ai_function(strata)

        if (isinstance(ei, (pd.Series | np.ndarray))) & (
            isinstance(hi, (pd.Series | np.ndarray))
        ):

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
        def _get_values(stratum: str, var: str) -> Any:
            """Get beta values from model for stratum."""
            return self.strata_results.get(stratum, {}).get(var, None)

        pop[f"{self.y_var}_beta"] = pop["_strata_var_mod"].apply(
            _get_values, var=f"{self.y_var}_beta"
        )

        # Calculate imputed values
        if self.method == "rate":
            pop_xvar: int | Any = pop[self.x_var]
        if self.method == "homogen":
            pop_xvar = 1

        pop[f"{self.y_var}_imp"] = pop[f"{self.y_var}_beta"] * pop_xvar

        # Remove beta? This is because it is based on sample x values so appears "wrong" for surprise strata.
        pop.drop(f"{self.y_var}_beta", axis=1, inplace=True)

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

        # function to map population and sample totals to survey data
        def _get_sums(stratum: str, var: str) -> Any:
            """Get sums within strata."""
            sum_value = self.strata_results.get(stratum, {}).get(var, None)
            return sum_value

        # Apply the function to get x values. Use strata_var_mod to consider surprise strata
        if self.method == "rate":
            sample_sum = utvalg["_strata_var_mod"].apply(
                _get_sums, var=f"{self.x_var}_sum_sample"
            )
            pop_sum = utvalg["_strata_var_mod"].apply(
                _get_sums, var=f"{self.x_var}_sum_pop"
            )

        # Use n and N instead for homogen
        if self.method == "homogen":
            sample_sum = utvalg["_strata_var_mod"].apply(_get_sums, var="n")
            pop_sum = utvalg["_strata_var_mod"].apply(_get_sums, var="N")

        utvalg["estimation_weights"] = pop_sum / sample_sum

        # Check for obs in suprise strata and set to 1 - should be 1
        mask = utvalg._strata_var_mod.str.contains("surprise")
        utvalg.loc[mask, "estimation_weights"] = 1

        return utvalg

    def _add_flag(self) -> None:
        """Add flag in population data to say if unit is in the sample or not."""
        self.flag_var: str = f"{self.y_var}_flag_sample"
        sample_ids = set(self.sample_data[self.id_nr])
        self.pop_data[self.flag_var] = self.pop_data[self.id_nr].apply(
            lambda x: 1 if x in sample_ids else 0
        )
