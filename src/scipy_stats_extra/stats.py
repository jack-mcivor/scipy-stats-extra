"""Interesting distributions. Consider as additional to `scipy.stats`
"""

import numpy as np
import scipy.special
import scipy.stats
import scipy.stats._boost as _boost


def _raw_binomial_pmf(x, n, p):
    # A faster version of `scipy.special.binom(n, x) * p**x * (1-p)**(n-x)`
    return _boost._binom_pdf(x, n, p)


class continuous_n_binomial_gen(scipy.stats.rv_discrete):
    """A version of the Binomial distribution where n is continuous, not integer
    This is achieved by replacing the Bionmial coefficient with the analogous Gamma functions
    The PMF has positive probability when k <= n
    """

    def _argcheck(self, n, p):
        return (n >= 0) & (p >= 0) & (p <= 1)

    @staticmethod
    def _legit_support_raw_pmf(x, n, p):
        _pmf = _raw_binomial_pmf(x, n, p)
        # _pmf[x > n] can be negative. It probably doesn't make sense to have "successes" > "trials" anyway
        _pmf[x > n] = 0
        return _pmf

    def _pmf(self, x, n, p):
        """
        require max(x) <= n
        """
        # HACK: the scipy machinery turns n & p into arrays, which messes with np.arange()
        # TODO: I think there is some way to vectorize this
        if isinstance(n, np.ndarray):
            n = n[0]
        if isinstance(p, np.ndarray):
            p = p[0]

        possible_xs = np.arange(0, np.floor(n) + 1)
        # TODO: this normalizing constant can be cached - just wrap in a function with @functools.cache
        normalizing_constant = self._legit_support_raw_pmf(possible_xs, n, p).sum()
        return self._legit_support_raw_pmf(x=x, n=n, p=p) / normalizing_constant


continuous_n_binomial = continuous_n_binomial_gen(name="continuous_n_binomial")


class binomial_rounding_mixture_gen(scipy.stats.rv_discrete):
    """A Binomial distribution with continuous n, formed as a mixture of the two Binomials with nearest integer n's"""

    def _argcheck(self, n, p):
        return (n >= 0) & (p >= 0) & (p <= 1)

    def _rvs(self, n, p, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        n_floor = np.floor(n)
        n_ceil = np.ceil(n)
        alpha = n - n_floor
        return np.where(
            random_state.random(size=size) < (1 - alpha),
            random_state.binomial(n=n_floor, p=p, size=size),
            random_state.binomial(n=n_ceil, p=p, size=size),
        )

    def _pmf(self, x, n, p):
        n_floor = np.floor(n)
        n_ceil = np.ceil(n)
        alpha = n - n_floor
        binom_down = scipy.stats.binom.pmf(n=n_floor, p=p, k=x)
        binom_up = scipy.stats.binom.pmf(n=n_ceil, p=p, k=x)
        return alpha * binom_up + (1 - alpha) * binom_down

    def _stats(self, n, p, moments="mv"):
        # let alpha = n - floor(n), n_floor = floor(n), n_ceil = ceil(n)

        # If n is integer, then alpha=0, n=n_floor=n_ceil and the variance reduces to the Binomial variance.
        mu = n * p  # same as the Binomial, even though n is allowed to be continuous
        n_floor = np.floor(n)
        n_ceil = np.ceil(n)
        alpha = n - n_floor
        var = (
            (1 - alpha) * n_floor * p * (1 - p)
            + alpha * n_ceil * p * (1 - p)
            + (1 - alpha) * (n_floor * p - n * p) ** 2
            + alpha * (n_ceil * p - n * p) ** 2
        )
        g1 = None  # TODO
        g2 = None  # TODO
        return mu, var, g1, g2


binomial_rounding_mixture = binomial_rounding_mixture_gen(name="binomial_rounding_mixture")


class com_poisson_gen(scipy.stats.rv_discrete):
    """The Conway-Maxwell-Poisson distribution (aka CMP or COM-Poisson)

    A Poisson distribution where the nu models a non-linear rate of decay in successive probabilities.
    nu=1 is the standard Poisson distribution.

    NB: The mean & variance use the sum of an infinite series. There are no closed form formulas.
    NB: someone generated this here https://github.com/scipy/scipy/issues/17620
    """

    def _argcheck(self, lam, nu):
        return (lam > 0) & (nu >= 0)

    def _pmf(self, x, lam, nu):
        _max_value = 100
        j = np.arange(_max_value)
        normalization_constant = np.sum(lam**j / (scipy.special.factorial(j) ** nu))
        return lam**x / (scipy.special.factorial(x) ** nu * normalization_constant)


com_poisson = com_poisson_gen(name="com_poisson")


class delaporte_gen(scipy.stats.rv_discrete):
    """The Delaporte distribution
    x ~ Poisson(lam + Gamma(a, b))

    A compound distribution based on a Poisson distribution, where there are two components to the mean parameter:
    * a fixed component, which has the mu parameter,
    * and a gamma-distributed variable component, which has the a and b parameters.

    When a & b = 0, the distribution is the Poisson
    When lam is 0, the distribution is the Gamma-Poisson (aka Negative Binomial)

    """

    def _argcheck(self, lam, a, b):
        return (lam >= 0) & (a >= 0) & (b >= 0)

    def _pmf(self, x, lam, a, b):
        raise NotImplementedError("Does not yet support vector x")
        # k = np.arange(x)
        # x: NDArray[int_]
        i = np.arange(x)[:, np.newaxis]
        return np.sum(
            (scipy.special.gamma(a + i) * b**i * lam ** (x - i) ** np.exp(-lam))
            / (
                scipy.special.gamma(a)
                * scipy.special.factorial(i)
                * (1 + b) ** (a + i)
                * scipy.special.factorial(x - i)
            ),
            axis=0,
        )

    def _stats(self, lam, a, b, moments="mv"):
        mu = lam + a * b
        var = lam + a * b * (1 + b)
        g1 = None
        g2 = None
        # fmt: off
        if "s" in moments:
            g1 = (lam + a*b*(1 + 3*b + 2*b**2)) / (lam + a*b*(1 + b))**(3 / 2)
        if "k" in moments:
            g2 = (
                lam + 3*lam**2 + a*b*(1 + 6*lam + 6*lam*b + 7*b + 12*b**2 + 6*b**3 + 3*a*b + 6*a*b**2 + 3*a*b**3)
            ) / (lam + a * b * (1 + b)) ** 2
        # fmt: on
        return mu, var, g1, g2

    def _rvs(self, lam, a, b, size=None, random_state=None):
        k = a
        # scipy defines scale = 1 / beta
        theta = b
        if size is None:
            size = 1
        assert random_state is not None
        return random_state.poisson(random_state.gamma(shape=k, scale=theta, size=size) + lam)


delaporte = delaporte_gen(name="delaporte")


class gamma_poisson_binomial_gen(scipy.stats.rv_discrete):
    """
    n ~ Poisson(Gamma(k, theta))
    x ~ Binomial(n, p)

    To re-parameterize in terms of alpha & beta of the Gamma distribution, use:
        alpha = k
        beta = 1/theta
    To re-parameterize in terms of n & p of the NegativeBinomial distribution, use:
        n = k (n is called r in the Wikipedia page)
        p = 1/(theta+1)
    """

    def _argcheck(self, pk, ptheta, p):
        return (pk > 0) & (ptheta > 0) & (p >= 0) & (p <= 1)

    # def _pmf(self, x, pk, ptheta, p):
    #     _max_value = 100
    #     i = np.arange(0, _max_value)[:, np.newaxis]
    #     return np.sum(
    #         scipy.stats.binom.pmf(k=x, n=i, p=p) * scipy.stats.nbinom.pmf(k=i, n=pk, p=1 / (ptheta + 1)), axis=0
    #     )

    def _rvs(self, pk, ptheta, p, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        # return random_state.binomial(n=random_state.poisson(random_state.gamma(shape=pk, scale=ptheta, size=size)), p=p)
        return random_state.binomial(n=random_state.negative_binomial(n=pk, p=1 / (ptheta + 1), size=size), p=p)

    def _stats(self, pk, ptheta, p, moments="mv"):
        mu = pk * ptheta * p
        var = p * pk * ptheta + p**2 * pk * ptheta**2
        g1 = None
        g2 = None
        return mu, var, g1, g2


gamma_poisson_binomial = gamma_poisson_binomial_gen(name="gamma_poisson_binomial")


class beta_poisson_binomial_gen(scipy.stats.rv_discrete):
    """
    p ~ Beta(alpha, beta)
    n ~ Poisson(lambda)
    x ~ Binomial(n, p)
    """

    def _argcheck(self, lam, a, b):
        return (lam > 0) & (a > 0) & (b > 0)

    def _pmf(self, x, lam, a, b):
        # TODO: need to integrate over the gamma distribution & sum over the poisson distribution
        raise NotImplementedError("Not complete")

    def _rvs(self, lam, a, b, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        return random_state.binomial(n=random_state.poisson(lam, size=size), p=random_state.beta(a=a, b=b, size=size))

    def _stats(self, lam, a, b, moments="mv"):
        mu = lam * a / (a + b)
        # fmt: off
        var = a**2*lam/(a+b)**2 + a*b/((a+b)**2*(a+b+1)) * ((a+b)*lam + lam**2 + lam)
        # fmt: on
        g1 = None
        g2 = None
        return mu, var, g1, g2


beta_poisson_binomial = beta_poisson_binomial_gen(name="beta_poisson_binomial")


class beta_gamma_poisson_binomial_gen(scipy.stats.rv_discrete):
    """
    p ~ Beta(alpha, beta)
    n ~ Poisson(Gamma(a, b))
    x ~ Binomial(n, p)
    """

    def _argcheck(self, an, bn, ap, bp):
        return (an > 0) & (bn > 0) & (ap > 0) & (ap > 0)

    def _pmf(self, x, an, bn, ap, bp):
        # TODO: need to integrate over the gamma distribution & sum over the gamma-poisson distribution
        raise NotImplementedError("Not complete")

    def _rvs(self, an, bn, ap, bp, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        return random_state.binomial(
            n=random_state.poisson(random_state.gamma(an, bn, size=size)), p=random_state.beta(a=ap, b=bp, size=size)
        )

    def _stats(self, an, bn, ap, bp, moments="mv"):
        raise NotImplementedError("Not complete")


beta_gamma_poisson_binomial = beta_gamma_poisson_binomial_gen(name="beta_gamma_poisson_binomial")


def _zero_inflated_mixture_moments(pi, base_dist_mean, base_dist_var):
    mu = (1 - pi) * base_dist_mean
    var = base_dist_mean**2 * (1 - pi) * pi - pi * base_dist_var + base_dist_var
    return mu, var


class zero_inflated_poisson_gen(scipy.stats.rv_discrete):
    def _argcheck(self, pi, lam):
        return (lam > 0) & (pi >= 0) & (pi <= 1)

    def _pmf(self, x, pi, lam):
        return np.where(x == 0, pi + (1 - pi) * np.exp(-lam), (1 - pi) * scipy.stats.poisson.pmf(mu=lam, k=x))

    def _rvs(self, pi, lam, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        return np.where(
            random_state.random(size=size) < pi,
            0,
            random_state.poisson(lam, size=size),
        )

    def _stats(self, pi, lam, moments="mv"):
        mu, var = _zero_inflated_mixture_moments(pi, base_dist_mean=lam, base_dist_var=lam)
        g1 = None
        g2 = None
        if "s" in moments:
            g1 = (2 * lam**2 * pi**2 - lam**2 * pi + 3 * lam * pi + 1) / (
                np.sqrt(lam) * (lam * pi + 1) * np.sqrt(-lam * pi**2 + lam * pi - pi + 1)
            )
        return mu, var, g1, g2


zero_inflated_poisson = zero_inflated_poisson_gen(name="zero_inflated_poisson")


class zero_inflated_nbinom_gen(scipy.stats.rv_discrete):
    def _argcheck(self, pi, n, p):
        return (n > 0) & (p > 0) & (p <= 1) & (pi >= 0) & (pi <= 1)

    def _pmf(self, x, pi, n, p):
        return np.where(
            x == 0,
            pi + (1 - pi) * p**n,  # p**n == scipy.stats.nbinom.pmf(n=n, p=p, k=0)
            (1 - pi) * scipy.stats.nbinom.pmf(n=n, p=p, k=x),
        )

    def _rvs(self, pi, n, p, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        return np.where(
            random_state.random(size=size) < pi,
            0,
            random_state.negative_binomial(n=n, p=p, size=size),
        )

    def _stats(self, pi, n, p, moments="mv"):
        base_dist_mean, base_dist_var = scipy.stats.nbinom.stats(n=n, p=p, moments="mv")
        mu, var = _zero_inflated_mixture_moments(pi, base_dist_mean=base_dist_mean, base_dist_var=base_dist_var)
        g1 = None
        g2 = None
        return mu, var, g1, g2


zero_inflated_nbinom = zero_inflated_nbinom_gen(name="zero_inflated_nbinom")


class hurdle_poisson_gen(scipy.stats.rv_discrete):
    def _argcheck(self, psi, lam):
        return (lam > 0) & (psi >= 0) & (psi <= 1)

    def _pmf(self, x, psi, lam):
        return np.where(x == 0, psi, (1 - psi) * scipy.stats.poisson.pmf(mu=lam, k=x) / (1 - np.exp(-lam)))

    def _rvs(self, psi, lam, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        is_zero = random_state.random(size=size) < psi
        non_zero = ~is_zero

        sample = np.zeros(size, dtype=np.int_)

        poisson_rvs = scipy.stats.poisson.rvs(lam, size=non_zero.sum())
        # Make sure we don't have any zeros from the Poisson sampling
        while np.any(poisson_rvs == 0):
            poisson_zeros = poisson_rvs == 0
            poisson_rvs[poisson_zeros] = scipy.stats.poisson.rvs(lam, size=poisson_zeros.sum())

        sample[non_zero] = poisson_rvs

        return sample

    def _stats(self, psi, lam, moments="mv"):
        mean_zero_trunc_pois = lam * np.exp(lam) / (np.exp(lam) - 1)
        var_zero_trunc_pois = mean_zero_trunc_pois * (1 + lam - mean_zero_trunc_pois)
        mean = (1 - psi) * mean_zero_trunc_pois
        var = (1 - psi) * (var_zero_trunc_pois + mean_zero_trunc_pois**2) - mean**2
        return mean, var, None, None


hurdle_poisson = hurdle_poisson_gen(name="hurdle_poisson")


class hurdle_nbinom_gen(scipy.stats.rv_discrete):
    def _argcheck(self, psi, n, p):
        return (n > 0) & (p > 0) & (p <= 1) & (psi >= 0) & (psi <= 1)

    def _pmf(self, x, psi, n, p):
        return np.where(x == 0, psi, (1 - psi) * scipy.stats.nbinom.pmf(n=n, p=p, k=x) / (1 - p**n))

    def _rvs(self, psi, n, p, size=None, random_state=None):
        if size is None:
            size = 1
        assert random_state is not None
        is_zero = random_state.random(size=size) < psi
        non_zero = ~is_zero

        sample = np.zeros(size, dtype=np.int_)

        nbinom_rvs = scipy.stats.nbinom.rvs(n=n, p=p, size=non_zero.sum())
        # Make sure we don't have any zeros from the nbinom sampling
        while np.any(nbinom_rvs == 0):
            nbinom_zeros = nbinom_rvs == 0
            nbinom_rvs[nbinom_zeros] = scipy.stats.nbinom.rvs(n=n, p=p, size=nbinom_zeros.sum())

        sample[non_zero] = nbinom_rvs

        return sample

    def _stats(self, psi, n, p, moments="mv"):
        mean_zero_trunc_nbinom = n * (p - 1) / (p * (p**n - 1))
        var_zero_trunc_nbinom = n * (p - 1) * (-n * (p - 1) + (p**n - 1) * (-n * p + n + 1)) / (p**2 * (p**n - 1) ** 2)
        mean = (1 - psi) * mean_zero_trunc_nbinom
        var = (1 - psi) * (var_zero_trunc_nbinom + mean_zero_trunc_nbinom**2) - mean**2
        return mean, var, None, None


hurdle_nbinom = hurdle_nbinom_gen(name="hurdle_nbinom")
