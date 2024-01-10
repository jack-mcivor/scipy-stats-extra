# SciPy Stats Extra
Implements some additional distributions and utilities on top of SciPy.

## Features
### Distributions
Currently has a number of interesting and compound distributions implemented.

| Distribution                        | Support                 | Parameters                                 |
|-------------------------------------|-------------------------|--------------------------------------------|
| Conway-Maxwell Poisson              | {0, 1, 2, ...}          | lam > 0, nu > 0                            |
| Delaporte                           | {0, 1, 2, ...}          | lam > 0, alpha > 0, beta > 0               |
| Gamma-Poisson-Binomial              | {0, ..., n}             | alpha > 0, beta > 0, 0 < p < 1             |
| Beta-Poisson-Binomial               | {0, ..., n}             | alpha > 0, beta > 0, lam > 0               |
| Beta-Gamma-Poisson-Binomial         | {0, ..., n}             | alpha > 0, beta > 0, a > 0, b > 0          |
| Zero Inflated Poisson               | {0, 1, 2, ...}          | lam > 0, pi ∈ [0, 1]                       |
| Zero Inflated Negative Binomial     | {0, 1, 2, ...}          | n > 0, p ∈ (0, 1], pi ∈ [0, 1]             |
| Hurdle Poisson                      | {0, 1, 2, ...}          | lam > 0, psi ∈ [0, 1]                      |
| Hurdle Negative Binomial            | {0, 1, 2, ...}          | n > 0, p ∈ (0, 1], psi ∈ [0, 1]            |
| "Continuous n" Binomial             | {0, ..., n}             | n > 0, p ∈ [0, 1]                          |
| "Mixture of nearest n" Binomial     | {0, ..., n}             | n > 0, p ∈ [0, 1]                          |

## Contributing
All contributions, questions or comments are welcome! Please open an issue or pull request.

### Setup
* Install [Poetry](https://python-poetry.org/docs/#installation)
* Install the package (as editable) with `poetry install`
* Install the pre-commit hooks with `pre-commit install`
* Run the tests with `pytest`
* Run the pre-commit hooks (linting, formatting, etc.) with `pre-commit run --all-files`

### Adding a new distribution
The best way to do this is to create a class that subclasses `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`. If the distribution is discrete, you must implement at least the `_pmf` or `_cdf` method. If the distribution is continuous, you must implement at least the `_pdf` or `_cdf` method. You should also implement the `_argcheck` method to check the validity of the parameters. This is required if the parameters are not constrained to be strictly positive (the default _argcheck implementation).

You can optionally implement the following methods `_logpdf/_logpmf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf, _munp, _stats`. By default these are imputed using the `_pdf` or `_pmf` method, but are more efficient if implemented directly. The default `_rvs` method uses inverse transform sampling via the `_ppf` (inverse-cdf) method.

For discrete distributions, instead of defining the pmf method, you can pass a list of x_k and p(x_k) values during instantiation.

Also see the [SciPy guide on this topic](https://docs.scipy.org/doc/scipy/tutorial/stats.html#building-specific-distributions) and the very helpful API documentation for [`scipy.stats.rv_continuous`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html) and [`scipy.stats.rv_discrete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html).

Each method should work with vectorized inputs, so that it may work with a grid of parameters, eg. produced by `np.meshgrid`.

A small example of implementing a discrete distribution:
```python
from scipy.stats import rv_discrete
class poisson_gen(rv_discrete):
    "Poisson distribution"
    def _pmf(self, k, mu):
        return exp(-mu) * mu**k / factorial(k)

poisson = poisson_gen(name="poisson")
```
This distribution can be used like any other SciPy distribution:
```python
# create a "frozen distribution" and sample from it
poisson_dist = poisson(mu=2)
poisson_dist.rvs(size=10)

# alternatively, use the class methods directly
poisson.rvs(mu=2, size=10)
```

To test that specific methods (eg. `_rvs` or `_stats`) is implemented correctly you can check:
* If `_rvs` is implemented, that the:
    * sample mean and variance match the theoretical values (`dist.mean()` and `dist.var()`)
    * sample value_counts matches the pmf (for discrete distributions)
