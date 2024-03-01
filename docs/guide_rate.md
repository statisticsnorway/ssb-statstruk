# Quick guide to running a rate model with **_statstruk_**

The python package **_statstruk_** is designed to make running estimation models for business surveys easier with options for standard and robust variance estimation as well as outlier detection. This quick guide is written (mostly) for those at Statistics Norway and uses an example data set, hosted on their internal Dapla platform. This is a practical guide with examples; see [Theory for rate model estimations](https://github.com/statisticsnorway/ssb-statstukrt/docs/theory_rate.md) for the formulas used in the functions.

## Installation
The package is on PyPI can be installed in a poetry environment by running the following in a terminal:
```bash
poetry install ssb-statstruk
```

## Import module
The module can then be imported into a python session/notebook. The main class used is called `ratemodel` and can be imported with:
```python
import ratemodel from statstruk
```

## Data requirements
The estimation models are design to run using a population data frame and a sample data frame. The **population** data frame should
- contain one row per unit (business).
- contain a unique identifier.
- contain a variable to use as the explanatory variable in the model (for example number of employees or turnover).
- may contain a variable(s) to use for stratification. This will create separate models within strata groups.

The sample data should:
- contain one row per unit (business) in the responding sample.
- contain a unique identifier which is the same as in the population data.
- contain one or more numerical, target variables which you are interested in estimating.
- may contain strata and explanatory variables, identical to that in the population file.

An example of population and sample data can be read in using the following code:

```python
import dapla as dp

bucket = "gs://ssb-prod-dapla-felles-data-delt"
folder = "felles/veiledning/python/eksempler/statstruk"

pop_df = dp.read_pandas(f"{bucket}/{folder}/pop.parquet")
sample_df = dp.read_pandas(f"{bucket}/{folder}/sample.parquet")
```

## Set up model
The model can be initialize with the ddata by running the ´ratemodel()´ function. This takes the names of the two data frames and the identification variable found in them both.

```python
mod = ratemodel(pop_df, sample_df, id_nr="id")
```
This function initiates an instance of the `ratemodel` class and runs some initial checks. If you want extra output throughout the fitting and esitmation you can set the paramter ´verbose=2´ for more descriptive inputs.

## Fit model
The model can now be fit using the `fit()` function. The name of the explanatory variable (`x_var`)  and target variable (`y_var`) need to be provided. If the model should be run within strata, the name of the variable(s) should be given as the `strata_var` parameter.

```python
mod.fit(x_var="employees", y_var="job_vacancies", strata_var="industry")
```
Several strata variables can be provided as a list. For example `strata_var=["industry", "size"]`:

## Get estimates
Estimates for the total values and their uncertainty within strata can be fetched with

```python
mod.get_estimates()
```
This calculates robust variance estimations and the coefficient of variation for each stratum. Formulas used in the estimation calculations can be found in [Theory for rate model estimations](https://github.com/statisticsnorway/ssb-statstukrt/docs/theory_rate.md). Sometimes it is useful to see the estimates that are not the strata but other domains (for example at a higher industry level, or for the country). This can be specified using the `domain` parameter.

```python
mod.get_estimates(domain = "country")
```
Domain groups may be either aggregates of the strata or other variables found in the population data. Robust variance estimates are only currently avaible for  domains that are aggregates of strata, otherwise standard variance estimates are calculated.

If you want to force standard variance estimations you can use the `variance_type="standard"` parameter.

## Dealing with outliers
Outliers can be identified using studentized residuals or $G$ values. They can be shown once a model is fitted using the `get_extremes()` function

```python
mod.get_extremes()
```
This will show observations that exceeded the studentized residuals or $G$ thresholds. Thresholds can be changed using the `rbound` (studentized residuals threshold) and `gbound` (g-value multiplier threshold) parameters.

```python
mod.get_extremes(gbound = 5, rbound = 5)
```

Specific outlier observation can also be removed from the fitting process using the `exclude` parameter. This should be a list of the id numbers of the units in the sample to remove. For example, to exclude units with id numbers "5" and "9":

```python
mod.fit(x_var="employees", y_var="job_vacancies", strata_var="industry", exclude = [5, 9])
```
These units are then classified into their own seperate stratum and don't contribute to the variance.

Alternativly, outliers can be automatically removed using the `exclude_auto` parameter. This specifies how many iterations to run the oulier detection and removal.

```python
mod.fit(x_var="employees", y_var="job_vacancies", strata_var = "industry", exclude_auto = 1)
```

The criteria for outlier removal can be specified using the `rbound` and `gbound` parameters.

## Speed up estimations
Outlier detection methods rely on fitting many models excluding one point at a time. This requires substantial runtime. If you do not need to control for outliers, the estimation can be performed and speed up by specifying `control_extremes = False`:
```python
mod.fit(x_var="employees", y_var="job_vacancies", strata_var = "industry", control_extremes=False)
```

## Get weights
Once the models are fit, weights can be extracted for the units in the sample. These can be used to create estimated poplation totals for target variables.
```python
mod.get_weights()
```

## Get imputations
Alternatively, target variables can be imputed for all observations in the population data. Those in the sample will have the observed values for the target variable.
```python
mod.get_imputed()
```
