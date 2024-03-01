# Theory for rate model estimations in **_statstruk_**

## Introduction
The **_statstruk_** package is based on standard statistical theory and that described in the Norwegian document: [Bruk av applikasjonen Struktur](https://www.ssb.no/a/publikasjoner/pdf/notat_200730/notat_200730.pdf). A summary of the theory used in programming **_statstruk_** is described here.

The package can be used to calculate model-based estimates for totals ($T$) of a target variable collected from sampled units ($s$) in a population ($U$). Standard and robust estimates for the uncertainty (variance) of the estimates are also calculated in the package. The methods provided are common for business surveys at Statistics Norway.

There are 3 common models used for estimating: rate, regression and homogenous models. Rate model estimation is describes here.

## Rate model estimation
Estimates based on a rate model are based on the following model:

$$
y_{hi} = \beta_h x_{hi} + \epsilon_{hi}
$$

where $y_{hi}$ is the variable of interest for unit $i$ in stratum $h$, $\beta_h$ is the rate for the stratum, $x_{hi}$ is the explanatory variable, and $epsilom_{hi}$ is the residual.

The variance structure of the residuals for a rate model is

$$
Var(\epsilon_{hi}) = x_{hi}\sigma^2_h
$$

### Total estimation
The value for $\beta_h$ is estimated using:

$$
\hat{\beta}_h = \frac{y_{s_h}}{x_{s_h}}
$$

where $y_{s_h}$ is the sum of the variable of interest for the sample in stratum $h$ and $x_{s_h}$ is the sum of the explanatory variables for the sampled units in stratum $h$.

The estimate for the total ($T_h$) in stratum $h$ is:

$$
\hat{T}_h = X_h \hat{\beta}_h
$$

where $X_h$ is the sum of the explanatory variable in stratum $h$ for the population.


### Standard variance estimation
A standard estimation for the uncertainty of the total estimate ($\hat{T_h}$), for a rate model, can be described as

$$
\hat{Var}(\hat{T}_h-T_h) = X_h^2 \frac{X_h-x_{s_h}}{X_h} \frac{\hat{\sigma}_h^2}{x_{s_h}}
$$

where $\hat{\sigma}_h^2$ is estimated from the model as

$$
\hat{\sigma}_h^2 =\frac{1}{n_h-1}\sum_{i\in s_h}\frac{(y_{hi} - \hat{\beta_h}x_{hi})^2}{x_{hi}}
$$

Furthermore, the standard error ($SE$) is

$$
SE(\hat{T_h}-T_h)= \sqrt{\hat{Var}(\hat{T_h}-T_h)}
$$

and the coefficient of variation ($CV$) is then

$$
CV(\hat{T_h}-T_h) = \frac{SE(\hat{T_h}-T_h)}{\hat{T_h}}
$$

### Robust variance estimation
The [Struktur application](https://www.ssb.no/a/publikasjoner/pdf/notat_200730/notat_200730.pdf) programmed in SAS also includes  robust variance estimation. Three of these are programmed in the **_statstruk_** package. These are summarized in this section.

We define the robust variance estimate for a rate model in two parts:

$$
\begin{align}
Var_{robust}(\hat{T}_h-T_h) &= Var(\sum_{i \notin s_h}\hat{y}_{hi}) + Var(\sum_{i \notin s_h}y_{hi}) \\
&=\frac{X_{h|s_h}}{x_{s_h}}^2 \sum_{i\in s_h}{d_{hi}}+ \frac{X_{h|s_h}}{x_{s_h}} \sum_{i\in s_h}{d_{hi}}
\end{align}
$$

where $X_{h|s_h}$ is the sum of the explanatory variable in stratum $h$ excluding the sum of those in the sample in that stratum ($s_h$). The variable $d_{hi}$ is then defined in the following ways:

$$
\begin{align}
Var_{robust1} &: d_{hi} = e_{hi}^2 \\
Var_{robust2} &: d_{hi} = \frac{e_{hi}^2}{1-hat_{hi}} \\
Var_{robust3} &: d_{hi} = \frac{e_{hi}^2}{(1-hat_{hi})^2}
\end{align}
$$

where $e_{hi}$ is the residual of observation $i$ in the model in stratum $h$ and $hat_{hi}$ is the $i$th value of the diagonal from hat matrix, defined for a rate model as

$$
hat_h=W_h^{1/2}X_h(X_h^TW_hX_h)^{−1}X_h^TW_h^{1/2}
$$

where $W_h$ is the vector of weights which is $1/X_h$ for a rate model.



### Domain estimation
We are often interested in estimating totals that are not only at the strata level. If domains are simply aggregated strata we can estimate the total for domain, $d$ as

$$
\hat{T}_d = \sum_{h \in d}\hat{T}_h
$$

Similarly the variance can also be summed to calculate the uncertainty of the domain estimates

$$
\hat{Var}(\hat{T_d}-T_d) = \sum_{h \in d}\hat{Var}(\hat{T_h}-T_h)
$$

If the domains are not aggregations of several strata, we need to adjust the estimates for the total and uncertainty to account for this. The estimate of the total ($T_{hd}$) for domain, $d$, in stratum, $h$, is

$$
\hat{T}_{hd} = X_{hd}\hat{\beta}_h
$$

and a total for the domain ($\hat{T}_d$) as

$$
\hat{T}_{d} = \sum_h \hat{T}_{hd}
$$

A standard variance for the domain estimate can be calculated by first the variance of the strata domains as

$$
\hat{Var}(\hat{T}_{hd} - T_{hd}) =  X_{hd|s_h}^2\frac{X_{hd|s_h} + x_{s_h}}{X_{hd|s_h}} \frac{\hat{\sigma}_h^2}{x_{s_h}}
$$

where $X_{hd|s_h}$ is the sum of the explanatory variable in the population for strata $h$, and domain $d$, excluding the $x$ values in the sample for that domain and stratum. The total for the domain is the sum over the strata as

$$
\hat{Var}(\hat{T}_{d} - T_{d}) = \sum_h \hat{Var}(\hat{T}_{hd} - T_{hd})
$$

## Outlier detection
Estimation based on models can be strongly influenced by outliers that have a strong influnce on the model estimates. The **_statstruk_** package provides two outlier detection metrics: studentized residual values and the difference of fits values (DFFITS or $G$).

### Studentized residuals
The studentized residuals ($t_{hi|i}$) are calculated using an estimate for $\sigma_h$ based on a fitted model without observation, $i$, and is sometimes referred to as the external studentized residuals. The studentized residuals are calculated as

$$
t_{hi|i} = \frac{y_{hi} - \hat{\beta_h}x_{hi}}{\sqrt{\hat{\sigma}_{h|i}^2x_{hi}}\sqrt{1-hat_{hi}}}
$$

where

$$
\hat{\sigma}_{h|i}^2 = \frac{1}{n_h-2}\sum_{j\ne i}\frac{(y_{hj}-\hat{\beta}_{h|i}x_{hj})^2}{x_{hj}}
$$

where $\hat{\beta}_{h|i}$ refers to the model estimate for the rate excluding observation $i$.
Absolute values of the studentized residual values above a criteria are then classified as outliers. A general threshold criteria used in **_statstruk_** is  2 but can be adjusted.

### DFFITS
The difference of fits ($G$) can be calculated from the studentized residuals and hat values as

$$
G=t_{hi|i} \sqrt{\frac{hats_{hi}}{1 - hats_{hi}}}
$$

Absolute values of $G$ above a specified threshold are classified as outliers. The thresold value used for outlier values of $G$ in rate models is generally $ \lambda \sqrt{1/n_h}$ where $\lambda$ is often set to 2 but can be adjusted.
