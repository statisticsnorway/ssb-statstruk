# # Code for rate estimation
# #### To do:
#
# - Add in standard variance (?)
# - Add in option for several y values.
# - Add in for small strata with warning (<3)
# - Document functions.
# - Write theory documentation with formulas in a markdown doc.
# - Write help file/instructs documentation
# - Write tests
# - Homogen model option
# - Regression model option
#
#

# +
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# -


class ratemodel(ssbmodel):
    """Class for estimating statistics for business surveys using a rate model"""

    def __init__(
        self, pop_data: pd.DataFrame, sample_data: pd.DataFrame, id_nr: str
    ) -> None:
        super().__init__(pop_data, sample_data, id_nr)
        self.pop_data = pop_data
        self.sample_data = sample_data

    def fit(
        self,
        y_var: str,
        x_var: str,
        strata_var: str = None,
        exclude: list[str] = None,
        exclude_auto: int = 0,
        quiet: bool = True,
    ) -> None:
        """Run and fit a rate model within strata

        Args:
            y_var: The target variable to estimate from the survey.
            x_var: The variable to use as the explanatory variable in the model.
            strata_var: The stratification variable.
            exclude: List of ID numbers for observations to exclude.
            exclude_auto: Whether extreme values should be automaticcally excluded from the models. Default 0. Integer 1 indicates extreme values should be reomved once and model run again.
            quiet: Whether to print details on the running of the model.

        Returns:
        """
        # Check variables
        self._check_variable(x_var, self.pop_data, data_name="population")
        self._check_variable(x_var, self.sample_data, remove_missing=True)
        self._check_variable(y_var, self.sample_data, remove_missing=True)
        self._check_variable(
            strata_var, self.pop_data, data_name="population", check_for_char=True
        )
        self._check_variable(strata_var, self.sample_data, check_for_char=True)

        # Define the model formula. For example, let's say you're modeling 'Number of Job Vacancies' as a function of 'Turnover'
        formula = y_var + "~" + x_var + "- 1"
        self.y_var = y_var
        self.x_var = x_var

        # Create stratum variables
        if (type(strata_var) == list) & (len(strata_var) > 1):
            self.sample_data["_stratum"] = self.sample_data[strata_var].apply(
                lambda x: "_".join(x.astype(str)), axis=1
            )
            self.pop_data["_stratum"] = self.pop_data[strata_var].apply(
                lambda x: "_".join(x.astype(str)), axis=1
            )
            self.strata_var = "_stratum"
        else:
            self.strata_var = strata_var

        # Check all strata in sample are in population
        unique_strata_pop = self.pop_data[self.strata_var].unique()
        unique_strata_sample = self.sample_data[self.strata_var].unique()

        all_values_present = all(
            value in unique_strata_sample for value in unique_strata_pop
        )
        assert (
            all_values_present
        ), "Not all strata in the population were found in the sample data. Please check"
        all_values_present = all(
            value in unique_strata_pop for value in unique_strata_sample
        )
        assert (
            all_values_present
        ), "Not all strata in the sample were found in the population data. Please check"

        # Create new strata variable for modelling in case there are excluded variables
        self.pop_data["_strata_var_mod"] = self.pop_data[self.strata_var]
        self.sample_data["_strata_var_mod"] = self.sample_data[self.strata_var]
        self.strata_var_mod = "_strata_var_mod"

        # Add in change of strata for those excluded from model as: "surprise_strata"
        def _update_strata(df, exclude):
            """Update files to include a new variable for modelling including suprise strata."""
            # Use the 'loc' method to locate the rows where ID is in the exclude list and update 'strata'
            mask = df[self.id_nr].isin(exclude)
            df.loc[mask, "_strata_var_mod"] = (
                df.loc[mask, "_strata_var_mod"]
                + "_surprise_"
                + df.loc[mask, self.id_nr].astype(str)
            )

        if exclude is not None:
            _update_strata(self.pop_data, exclude)
            _update_strata(self.sample_data, exclude)

        def _get_hat(X, W):
            """Get the hat matrix for the model."""
            # Compute the square root of the weight matrix, W^(1/2)
            W = np.diag(W)  # Convert to n x n matrice
            W_sqrt = np.sqrt(W)

            # Compute (X^T * W * X)^(-1)
            XTWX_inv = np.linalg.inv(X.T @ W @ X)

            # Compute the hat matrix
            H = W_sqrt @ X @ XTWX_inv @ X.T @ W_sqrt

            return np.diag(H)  # Return diagonal

        def _get_rstud(y, res, x_var, df, hh, X, formula):
            """Get the studentized residuals from the model"""
            # set up vectors
            n = len(y)
            y = np.array(y)
            X = np.array(X)
            beta_ex_values = np.zeros(n)
            sigma2_values = np.zeros(n)
            R = np.zeros(n)

            for i in range(n):

                # Exclude the i-th observation
                df_i = df.drop(index=df.iloc[i].name)
                ww_i = 1 / (df_i[x_var])
                ww_i.loc[df_i[x_var] == 0] = 1 / (df_i[x_var] + 1)
                X_i = np.delete(X, i, axis=0)
                y_i = np.delete(y, i, axis=0)

                # Fit the WLS model without the i-th observation
                model_i = smf.wls(formula, data=df_i, weights=ww_i).fit()
                beta_ex_values[i] = model_i.params[x_var].item()

                # get predicted values y_j
                y_hat_j = model_i.predict(df_i)

                # Calculate sigma
                sigma2 = sum((y_i - y_hat_j) ** 2 / X_i) * 1 / (n - 2)

                # Calculate and save studentized residuals
                R[i] = res[i] / (np.sqrt(sigma2 * X[i]) * np.sqrt(1 - hh[i]))

            return (R, beta_ex_values)

        # Set up coefficient dictionaries
        strata_results = {}  # Each stratum as key
        obs_data = {}  # Each stratum as key - consider changing to virk id?

        # Iterate over each stratum in sample and fit model
        for stratum, group in self.sample_data.groupby(self.strata_var_mod):
            if len(group) > 1:  # Ensure there is more than one row to fit a model
                if not quiet:
                    print(f"\nFitting model for Stratum: {stratum}")

                # Adjusting weights for zero values in 'Number of Employees'
                weights = 1 / (group[x_var])
                weights.loc[group[x_var] == 0] = 1 / (group[x_var] + 1)

                # Fit the weighted least squares model
                model = smf.wls(formula, data=group, weights=weights).fit()
                sigma2 = (1 / (len(group) - 1)) * sum(
                    model.resid.values**2 / group[x_var]
                )
                stratum_info = {
                    self.strata_var_mod: stratum,
                    "beta": model.params[x_var].item(),  # Series of coefficients
                    "n": len(group),  # Number of observations in the sample
                    "x_sum_sample": group[x_var].sum(),
                    "sigma2": sigma2,
                }

                # add in condition for strata of two here?
                hats = _get_hat(group[[x_var]].values, weights)
                rstuds = _get_rstud(
                    y=group[y_var],
                    res=model.resid.values,
                    x_var=x_var,
                    df=group,
                    hh=hats,
                    X=group[x_var],
                    formula=formula,
                )

                # Create dict with observation level data
                obs_info = {
                    self.strata_var_mod: stratum,
                    self.id_nr: group[self.id_nr].values,
                    "xvar": group[x_var],
                    "yvar": group[y_var],
                    "resids": model.resid.values,
                    "hat": hats,
                    "rstud": rstuds[0],
                    "G": rstuds[0] * (np.sqrt(hats / (1 - hats))),
                    "beta_ex": rstuds[1],
                    #'threshold':influence.dffits[1]
                }

            else:
                if "surprise" not in stratum:
                    print(
                        f"\nStratum: {stratum}, has only one observation and has 0 variance. Consider combing strata."
                    )
                stratum_info = {
                    self.strata_var_mod: stratum,
                    "beta": (group[y_var].values / group[x_var].values)[0],
                    "n": len(group),  # Number of observations in the sample
                    "x_sum_sample": group[x_var].sum(),
                    "sigma2": 0,
                }
                obs_info = {
                    self.strata_var_mod: stratum,
                    self.id_nr: group[self.id_nr].values,
                    "xvar": group[x_var],
                    "yvar": group[y_var],
                    "resids": [0],
                    "hat": np.nan,
                    "rstud": np.nan,
                    "G": np.nan,
                    "beta_ex": np.nan,
                }
            strata_results[stratum] = stratum_info
            obs_data[stratum] = obs_info

        # Loop through population also
        for stratum, group in self.pop_data.groupby(self.strata_var_mod):
            stratum_info = {"N": len(group), "x_sum_pop": group[x_var].sum()}
            strata_results[stratum].update(stratum_info)

        # Set results to instance
        self.strata_results = strata_results
        self.obs_data = obs_data

        # Check for outliers and re-run if auto exclude is on
        if exclude_auto > 0:
            extremes = self.get_extremes()[self.id_nr].values.tolist()
            print(extremes)
            print(type(extremes))
            if exclude is None:
                exclude = extremes
            else:
                exclude = exclude + extremes
            exclude_auto -= 1

            self.fit(y_var, x_var, strata_var, exclude, exclude_auto, quiet)

    @property
    def get_coeffs(self) -> pd.DataFrame:
        """Get the model coefficients for each strata"""
        return pd.DataFrame(self.strata_results).T

    @property
    def get_obs(self) -> pd.DataFrame:
        """Get the details for observations from the model"""
        return self.obs_data

    def _get_robust(self, strata: str):
        """Get robust variance estimations"""
        # collect data for strata
        x_pop = self.strata_results[strata]["x_sum_pop"]
        x_utv = self.strata_results[strata]["x_sum_sample"]
        hi = self.obs_data[strata]["hat"]
        ei = self.obs_data[strata]["resids"]
        if len(ei) == 1:
            return (0, 0, 0)

        # Calculate ai
        Xr = x_pop - x_utv
        ai = Xr / x_utv

        # Calculate di variations
        di_1 = ei**2
        di_2 = ei**2 / (1 - hi)
        di_3 = ei**2 / ((1 - hi) ** 2)

        # Caluclate variances
        V1 = sum(ai**2 * di_1) + sum(di_1) * ai
        V2 = sum(ai**2 * di_2) + sum(di_2) * ai
        V3 = sum(ai**2 * di_3) + sum(di_3) * ai
        return (V1, V2, V3)

    def _get_domain_estimates(self, domain: str):
        """Get domain estimation for case where domains are not an aggregation of strata."""
        # Collect data
        self._add_flag()  # add in check for if this is done?
        pop = self.get_imputed()
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
            est = np.sum(temp_dom[f"{self.y_var}_imputed"])  # for 1-y varible only

            # Loop through strata to get the partial variances
            var = 0
            for s in strata_unique:
                mask_s = (temp_dom[strata_var] == s) & (
                    temp_dom[self.flag_var] == 0
                )  # Those not in sample
                Uh_sh = np.sum(temp_dom.loc[mask_s, self.x_var])
                xh = res[s]["x_sum_sample"]
                s2 = res[s]["sigma2"]
                var += s2 * (Uh_sh + xh) / xh * Uh_sh

            # Add calculations to domain dict
            domain_df[d] = {
                "domain": d,
                "N": N,
                "n": n,
                f"{self.y_var}_est": est,
                f"{self.y_var}_variance": var,
                f"{self.y_var}_CV": np.sqrt(var) / est * 100,
            }

        # Format and return
        domain_df = pd.DataFrame(domain_df).T
        return domain_df

    def _get_domain(self, domain: str) -> pd.DataFrame:
        """Get mapping of domain to the strata results"""
        strata_var = self.strata_var_mod

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
        domain_mapped = strata_res[self.strata_var_mod].map(domain_key)

        return domain_mapped

    def get_estimates(
        self, domain: str = None, var_type: str = "robust"
    ) -> pd.DataFrame:
        """Get estimates for previously run model within strata or domains. Variance and CV estimates are returned for each domain.

        Args:
            domain: Name of the variable to use for estimation. Should be in the population data.
            var_type: Choose from 'robust' or 'standard' estimation of variance. Currently only robust estimation is calculated for strata and aggregated strata domains estimation and standard for other domains.

        Return:
            A pd.Dataframe is returned conatining estimates and variance/coefficient of variation estimations for each domain.

        """
        # Check model is run
        self._check_model_run()

        # Fetch results
        strata_df = pd.DataFrame(self.strata_results).T
        obs_df = pd.DataFrame(self.obs_data).T

        # Add in domain
        if domain is None:
            domain = self.strata_var
        try:
            strata_df[domain] = self._get_domain(domain).str[0]

        # If domains are not aggregates of strata run alternative calculation for variance (not robust)
        except:
            if var_type == "robust":
                print(
                    "Domain variable is not an aggregation of strata variables. Only standard variance calculations are available"
                )
            return self._get_domain_estimates(domain)

        # Format variables
        strata_df["N"] = pd.to_numeric(strata_df["N"])
        strata_df["n"] = pd.to_numeric(strata_df["n"])

        strata_df["x_sum_sample"] = pd.to_numeric(strata_df["x_sum_sample"])
        strata_df["x_sum_pop"] = pd.to_numeric(strata_df["x_sum_pop"])

        # Add estimates
        strata_df["beta"] = pd.to_numeric(strata_df["beta"])
        strata_df[self.y_var + "_est"] = pd.to_numeric(
            strata_df["beta"] * strata_df["x_sum_pop"]
        )

        # Add variance
        var1 = []
        var2 = []
        var3 = []
        for i, s in enumerate(strata_df[self.strata_var_mod]):
            var = self._get_robust(s)
            var1.append(var[0])
            var2.append(var[1])
            var3.append(var[2])

        strata_df["var1"] = np.array(var1)
        strata_df["var2"] = np.array(var2)
        strata_df["var3"] = np.array(var3)

        # Aggregate to domain
        result = (
            strata_df[[domain, "N", "n", "job_vacancies_est", "var1", "var2", "var3"]]
            .groupby(domain)
            .sum()
        )

        # Add in CV
        result[self.y_var + "_CV1"] = (
            np.sqrt(result.var1) / result[self.y_var + "_est"] * 100
        )
        result[self.y_var + "_CV2"] = (
            np.sqrt(result.var2) / result[self.y_var + "_est"] * 100
        )
        result[self.y_var + "_CV3"] = (
            np.sqrt(result.var3) / result[self.y_var + "_est"] * 100
        )

        # drop extra variables
        result = result.drop(["var1", "var2", "var3"], axis=1)

        return result

    def get_extremes(self, rbound: float = 2, gbound: float = 2) -> pd.DataFrame:
        """Get observations with extreme values based on their rstudized residual value or G value.

        Args:
            rbound: Multiplicative value to determine the extremity of the studentized residual values.
            gbound: Multiplicative value to determine the extremity of the G values.

        Return:
            A pd.DataFrame containing units with extreme values beyond a set boundary.
        """
        self._check_model_run()

        extremes = pd.DataFrame()
        for k in self.get_obs.keys():
            new = pd.DataFrame.from_dict(self.get_obs[k])
            new["beta"] = self.strata_results[k]["beta"]
            new[self.strata_var_mod] = self.strata_results[k][self.strata_var_mod]
            new["n"] = self.strata_results[k]["n"]
            new["N"] = self.strata_results[k]["N"]
            new[self.x_var] = new["xvar"]
            new[self.y_var] = new["yvar"]
            new["x_sum_pop"] = self.strata_results[k]["x_sum_pop"]
            new["y_est"] = new["beta"] * new["x_sum_pop"]
            new["y_est_ex"] = new["beta_ex"] * new["x_sum_pop"]
            new["gbound"] = gbound * np.sqrt(1 / self.strata_results[k]["n"])
            extremes = pd.concat([extremes, new])
        condr = np.abs(extremes["rstud"]) > rbound
        condg = np.abs(extremes["G"]) > extremes["gbound"]
        extremes = extremes.loc[condr | condg]
        extremes = extremes[
            [
                self.id_nr,
                self.strata_var_mod,
                "n",
                "N",
                self.x_var,
                self.y_var,
                "y_est",
                "y_est_ex",
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
        def _get_beta(stratum):
            return self.strata_results.get(stratum, {}).get("beta", None)

        pop["beta"] = pop[self.strata_var_mod].apply(_get_beta)

        # Calculate imputed values
        pop[f"{self.y_var}_imputed"] = pop["beta"] * pop[self.x_var]

        # Link in survey values
        id_to_yvar_map = utvalg.set_index(self.id_nr)[self.y_var]
        pop[f"{self.y_var}_imputed"] = (
            pop[self.id_nr].map(id_to_yvar_map).fillna(pop[f"{self.y_var}_imputed"])
        )

        return pop

    def get_weights(self) -> pd.DataFrame:
        """Get sample data with weights based on model.

        Returns:
            Pandas data frame with sample data and weights.
        """
        self._check_model_run()

        utvalg = self.sample_data

        # map population and sample x totals to survey data
        def get_sums(stratum, var):
            return self.strata_results.get(stratum, {}).get(var, None)

        # Apply the function to create a new 'beta' column in the DataFrame. Use strata_var_mod to consider surprise strata
        sample_sum = utvalg[self.strata_var_mod].apply(get_sums, var="x_sum_sample")
        pop_sum = utvalg[self.strata_var_mod].apply(get_sums, var="x_sum_pop")

        utvalg["estimation_weights"] = pop_sum / sample_sum

        return utvalg
