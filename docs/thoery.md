# Theory for estimations in **statstrukt**

## Introduction
The **statstrukt** package is based on standard statistical theory and that described in the Norwegian document: [Bruk av applikasjonen Struktur](https://www.ssb.no/a/publikasjoner/pdf/notat_200730/notat_200730.pdf). A summary of the theory used in programming **statsstrukt** is described here.

The package can be used to calculate model-based estimates for totals ($T$) of a target variable collected for some sampled units ($s$) in a population ($U$). Standard and robust estimates for the uncertainty (variance) of the estimates are also calculated in the package. The methods provided are common for

There are 3 common models used for estimating: rate, regression and homogenous models. These are described in the following sections

## Rate model
Estimates based on a rate model are based on the following model:
$$
y_hi = \beta_h x_{hi} + \epsilon_{hi}
$$
where $y_i$ is the variable of interest for unit $i$ in stratum $h$, $\beta_h$ is the rate for the stratum, $x_{hi}$ is the explanatory variable, and $epsilom_{hi}$ is the residual.

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

The estimate for the total in stratum $h$ ($T_h$)is:
$$
\hat{T}_h = X_h \hat{\beta}_h
$$
where $X_h$ is the sum of the explanatory variable in stratum $h$ for the population.


### Standard variance estimation
A standard estimation for the uncertainty of the total estimate ($\hat{T_h}$) can be described as
$$
\hat{Var}(\hat{T}_h-T_h) = X_h^2 \frac{X_h-x_{s_h}}{X_h} \frac{\hat{\sigma}^2}{x_{s_h}}
$$
where $\hat{\sigma^2}$ is estimated from the model as
$$
\hat{\sigma}^2 =\frac{1}{n_h-1}\sum_{i\in s_h}\frac{(y_{hi} - \hat{\beta_h}x_{hi})^2}{h_{hi}}
$$
Furthermore, the standard error ($SE$) is

$$
SE(\hat{T_h}-T_h)= \sqrt{\hat{Var}(\hat{T_h}-T_h)}
$$
and the coefficient of variation ($CV$) is
$$
CV(\hat{T_h}-T_h) = \frac{SE(\hat{T_h}-T_h)}{\hat{T_h}}
$$

### Robust estimation
The [Struktur application](https://www.ssb.no/a/publikasjoner/pdf/notat_200730/notat_200730.pdf) programmed in SAS contained estimations for robust variance. Three of these are programmed in the **statstrukt** package.

More coming soon...


### Domain estimation
We are often interested in estimating totals that are not only at the strata level. If domains are simply aggregated strata we can estimate the total for domain, $d$ as
$$
\hat{T}_d = \sum_{h \in d}\hat{T}_h
$$
Similarly the variance can also be summed to calculate the uncertainty of teh domain estimates
$$
\hat{Var}(\hat{T_d}-T_d) = \sum_{h \in d}\hat{Var}(\hat{T_h}-T_h)
$$

If the domains are not aggregations of several strata, we need to adjust the estimates for the total and uncertainty to account for this. The estimate of the total ($T_hd$) for domain, $d$, in stratum, $s$, is
$$
\hat{T}_{hd} = X_{hd}\hat{\beta}_h
$$
and a total for the domain ($\hat{T}_d$) as
$$
\hat{T}_{d} = \sum_h \hat{T}_{hd}
$$
A standard variance for the domain estimate can be calculated by first the variance of the strata domains as
$$
\hat{Var}(\hat{T}_{hd} - T_{hd}) = \hat{\sigma}_h^2 \frac{X_{hd|s_h} + x_{s_h}}{x_{s_h}} X_{hd|s_h}
$$
where $X_{hd|sh}$ is the sum of the xplanatory variable in the population for strata $h$, and domain $d$, excluding the $x$ values in the sample for that domain and stratum. The total for the domain is the sum over the strata as
$$
\hat{Var}(\hat{T}_{d} - T_{d}) = \sum_h \hat{Var}(\hat{T_{hd}} - T_{hd})
$$
