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
        
    def fit(
        self,
        y_var: str,
        x_var: str="",
        strata_var: str | list[str] = "",
        control_extremes: bool = True,
        exclude: list[str | int] = [],
        remove_missing: bool = True,
        rbound: float = 2,
        gbound: float = 2,
        method: str = "rate"
    ) -> None:
        """Run and fit a rate model within strata.

        Args:
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
        #self._check_variable(x_var, self.pop_data, data_name="population")               # move to ratemodel
        #self._check_variable(x_var, self.sample_data, remove_missing=remove_missing)    # move to ratemodel
        self._check_variable(y_var, self.sample_data, remove_missing=remove_missing)

        # Swap dtype if Int64 or Float64 to int64 or float64 (since these don't work with some of the models
        #self.pop_data[x_var] = self._convert_var(x_var, self.pop_data)                  # move to ratemodel
        #self.sample_data[x_var] = self._convert_var(x_var, self.sample_data)            # move to ratemodel
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
        
        # Add x=0 units in sample to exclude list and check y=0 for these
        #exclude = self._exclude_zeros(exclude)                                             #ratemodel
        
        # Update for strata_var_mod to surprise strata for those in exclude list
        self.pop_data = self._update_strata(self.pop_data, exclude)
        self.sample_data = self._update_strata(self.sample_data, exclude)
        
        
        # Set up coefficient dictionaries
        strata_results: dict[str, Any] = {}  # Each stratum as key
        obs_data: dict[str, Any] = {}  # Each stratum as key
        one_nonzero_strata: list[Any] = []                                                   #ratemodel
        
        # Iterate over each stratum in sample and fit model
        for stratum, group in self.sample_data.groupby("_strata_var_mod"):
            stratum_info, obs_info, one_nonzero = self._fit_model_and_controls(
                stratum, group
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
                stratum_info.update({"x_sum_pop": group[x_var].sum()})
                
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
        self.pop_data.loc[:,"_strata_var_mod"] = self.pop_data[strata_var_new].copy()
        self.sample_data.loc[:,"_strata_var_mod"] = self.sample_data[strata_var_new].copy()

        return strata_var_new
   
    def _check_strata(
        self, strata_var_new: str, exclude: list[str | int]
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

    def _exclude_zeros(self, exclude: list[str | int]) -> list[str | int]:
        """Check for observations with x=0 and move to exclude list. Check that all have y_var as 0."""
        mask0 = self.sample_data[self.x_var] == 0
        zeroysum = self.sample_data.loc[mask0, self.y_var].sum()
        assert (
            zeroysum == 0
        ), f"There are observations in your sample where {self.x_var} is zero but {self.y_var} is > 0. This is not allowed in a rate model. Please adjust or remove them."

        # Add to exclude list if doesn't fail
        if mask0.sum() > 0:
            print(
                f'There are {mask0.sum()} observations in the sample with {self.x_var} = 0. These are moved to "surprise strata".'
            )

            exclude = exclude + self.sample_data.loc[mask0, self.id_nr].tolist()

        return exclude

    def _update_strata(
        self, df: pd.DataFrame, exclude: list[str | int]
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
        """Get the studentized residuals from the model."""
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
        self,
        stratum: Any,
        group: pd.DataFrame,
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
            #stratum_info.update("x_sum_sample": group[x_var].sum())
            obs_info.update({"xvar": group[self.x_var]})

        if len(group) > 1:  # Ensure there is more than one row to fit a model
            if self.verbose == 2:
                print(f"\nFitting model for Stratum: {stratum!r}")
                
            if self.method == "rate":
                stratum_info, obs_info = self._rate_method(stratum, group, stratum_info, obs_info)
                
            elif self.method == "homogen":
                stratum_info, obs_info = self._homogen_method(stratum, group, stratum_info, obs_info)

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
                
             # Set x=1 so doesn't produce error. beta will still be 0 for x=0 as y must = 0 for rate model - !!need to check for 
            if (not x) | (group[self.x_var].values[0] == 0):
                x = 1 
            else:
                # Add standard info in for 1 obs strata : check for x-values = 0
                x = group[self.x_var].values[0]
                
            stratum_info.update(
                {
                    "beta": group[self.y_var].values[0] / x,
                    "sigma2": 0,
                }
            )
            obs_info.update({"resids": [0], "hat": np.nan})
            if self.control_extremes:
                obs_info.update({"rstud": np.nan, "G": np.nan, "beta_ex": np.nan})

        return stratum_info, obs_info, one_nonzero_strata
    
    
    def _rate_method(self, stratum, group, stratum_info, obs_info):
        stratum_info.update({"x_sum_sample": group[self.x_var].sum()})
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
                "beta": model.params[self.x_var].item(),  # Series of coefficients
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
                obs_info.update({"rstud": np.nan, "G": np.nan, "beta_ex": np.nan})
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
                        "beta_ex": rstuds[1],
                    }
                )
        
        return stratum_info, obs_info
        
        
        
    def _homogen_method(self, stratum, group, stratum_info, obs_info):
        
        formula = self.y_var + "~ 1"
        
        # Fit model
        model = smf.ols(formula, data=group).fit()
        sigma2 = model.scale
        stratum_info.update(
            {
                "beta": model.params['Intercept'].item(),
                "sigma2": sigma2,
            }
        )
        
        obs_info.update({"resids": model.resid.values})
        
        if self.control_extremes:
            if len(group) == 2:
                print(
                    f"Extreme values not able to be detected in stratum: {stratum!r} due to too few observations."
                )
                obs_info.update({"rstud": np.nan, "G": np.nan, "beta_ex": np.nan})
            else:
                influence = model.get_influence()
                hats = influence.hat_matrix_diag
                rstuds = model.get_influence().get_resid_studentized_external()
                obs_info.update({"resids": model.resid.values, "hat": hats,"rstud": rstuds,  
                                 "G": rstuds * (np.sqrt(hats / (1.0 - hats))), "beta_ex": np.nan})
        
        return stratum_info, obs_info
    
    
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
            assert False, "Standard variance not programmed yet" # temp!!!
            #return self._get_domain_estimates(domain, uncertainty_type) # Need to check and develop this

        # Format variables
        strata_df["N"] = pd.to_numeric(strata_df["N"])
        strata_df["n"] = pd.to_numeric(strata_df["n"])
        strata_df["beta"] = pd.to_numeric(strata_df["beta"])
        
        if self.method == "rate":
            strata_df[f"{self.x_var}_sum_pop"] = pd.to_numeric(strata_df["x_sum_pop"])
            strata_df[f"{self.x_var}_sum_sample"] = pd.to_numeric(strata_df["x_sum_sample"])
            x_pop = strata_df[f"{self.x_var}_sum_pop"]
        elif self.method == "homogen":
            x_pop = strata_df["N"]

        # Add estimates
        strata_df[f"{self.y_var}_EST"] = pd.to_numeric(
            strata_df["beta"] * x_pop
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
            assert False, "robust not calculated yet" # temp !!
            var1 = []
            var2 = []
            var3 = []
            for s in strata_df["_strata_var_mod"]:
                var = self._get_robust(s) ### Need to do this !!
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
    
        
    def _get_standard_variance(self, strata: str) -> Any:
        """Get standard variance estimates."""
        s2 = self.strata_results[strata]["sigma2"]
        if self.method == "rate":
            x_pop = self.strata_results[strata]["x_sum_pop"]
            x_utv = self.strata_results[strata]["x_sum_sample"]

        elif self.method == "homogen":
            x_pop = self.strata_results[strata]["N"] # Here we use counts instead of x
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