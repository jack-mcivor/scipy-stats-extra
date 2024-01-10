import numpy as np
import scipy
from scipy_stats_extra.stats import (
    beta_gamma_poisson_binomial,
    beta_poisson_binomial,
    binomial_rounding_mixture,
    com_poisson,
    continuous_n_binomial,
    delaporte,
    gamma_poisson_binomial,
    hurdle_nbinom,
    hurdle_poisson,
    zero_inflated_nbinom,
    zero_inflated_poisson,
)


def test_hurdle_poisson():
    dist = hurdle_poisson(psi=0.12, lam=4.5)
    m, v = dist.stats(moments="mv")
    print(f"HurdlePoisson(psi=0.12, lam=10.5) mean: {m:.2f}, var: {v:.2f}")
    assert -1e-2 < dist.pmf(0) - 0.12 < 1e-2
    assert -1e-2 < dist.pmf(np.arange(30)).sum() - 1 < 1e-2
    sample = dist.rvs(size=1_000_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    np.testing.assert_allclose(counts / 1_000_000, dist.pmf(vals), atol=1e-3)
    # check the mean & variance agree with the pmf & sample
    assert -1e-1 < (dist.pmf(vals) * vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1


def test_hurdle_nbinom():
    dist = hurdle_nbinom(psi=0.22, n=4.5, p=0.4)
    m, v = dist.stats(moments="mv")
    print(f"Hurdlenbinom(psi=0.12, lam=10.5) mean: {m:.2f}, var: {v:.2f}")
    assert -1e-2 < dist.pmf(0) - 0.22 < 1e-2
    assert -1e-2 < dist.pmf(np.arange(30)).sum() - 1 < 1e-2
    sample = dist.rvs(size=1_000_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    np.testing.assert_allclose(counts / 1_000_000, dist.pmf(vals), atol=1e-3)
    # check the mean & variance agree with the pmf & sample
    assert -1e-1 < (dist.pmf(vals) * vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1


def test_zero_inflated_poisson():
    dist = zero_inflated_poisson(pi=0.12, lam=10.5)
    m, v, s = dist.stats(moments="mvs")
    print(f"ZIPoisson(pi=0.12, lam=10.5) mean: {m:.2f}, var: {v:.2f}, skw: {s:.2f}")
    assert -1e-2 < dist.pmf(np.arange(30)).sum() - 1 < 1e-2
    sample = dist.rvs(size=1_000_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    np.testing.assert_allclose(counts / 1_000_000, dist.pmf(vals), atol=1e-3)
    # check the mean & variance agree with the pmf & sample
    assert -1e-1 < (dist.pmf(vals) * vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1
    assert -1e-1 < scipy.stats.skew(sample) - s < 1e-1


def test_zero_inflated_nbinom():
    dist = zero_inflated_nbinom(pi=0.12, n=10.5, p=0.7)
    m, v = dist.mean(), dist.var()
    print(f"ZINBinom(pi=0.12, n=10.5, p=0.2) mean: {m:.2f}, var: {v:.2f}")
    assert -1e-2 < dist.pmf(np.arange(100)).sum() - 1 < 1e-2
    sample = dist.rvs(size=100_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    np.testing.assert_allclose(counts / 100_000, dist.pmf(vals), atol=1e-2)
    # check the mean & variance agree with the pmf & sample
    assert -1e-1 < (dist.pmf(vals) * vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1


def test_gamma_poisson_binomial():
    dist = gamma_poisson_binomial(pk=10.5, ptheta=0.4, p=0.8)
    m, v = dist.mean(), dist.var()
    print(f"GammaPoissonBinom(k=10.5, theta=0.4, p=0.8) mean: {m:.2f}, var: {v:.2f}")
    assert -1e-2 < dist.pmf(np.arange(50)).sum() - 1 < 1e-2
    sample = dist.rvs(size=1_000_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    np.testing.assert_allclose(counts / 1_000_000, dist.pmf(vals), atol=1e-3)
    # check the mean & variance agree with the pmf & sample
    assert -1e-1 < (dist.pmf(vals) * vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1


def test_beta_poisson_binomial():
    dist = beta_poisson_binomial(lam=10.5, a=9.5, b=8.5)
    m, v = dist.mean(), dist.var()
    print(f"BetaPoissonBinom(lam=10.5, a=9.5, b=8.5) mean: {m:.2f}, var: {v:.2f}")
    # TODO: The PMF
    # assert -1e-2 < dist.pmf(np.arange(20)).sum() - 1 < 1e-2
    sample = dist.rvs(size=100_000)
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1


def test_beta_gamma_poisson_binomial():
    dist = beta_gamma_poisson_binomial(an=10.5, bn=5.5, ap=9.5, bp=8.5)
    # TODO: everything
    print("BetaGammaPoissonBinom(an=10.5, bn=5.5, ap=9.5, bp=8.5)")
    _sample = dist.rvs(size=100_000)


def test_continuous_n_binomial():
    dist = continuous_n_binomial(n=10.5, p=0.4)
    m, v = dist.mean(), dist.var()
    print(f"ContNBinom(n=10.5, p=0.4) mean: {m:.2f}, var: {v:.2f}")
    # TODO: The PMF is slow
    assert -1e-2 < dist.pmf(np.arange(20)).sum() - 1 < 1e-2
    sample = dist.rvs(size=100_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    np.testing.assert_allclose(counts / 100_000, dist.pmf(vals), atol=1e-2)
    # check the mean & variance agree with the pmf & sample
    assert -1e-1 < (dist.pmf(vals) * vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1


def test_binomial_rounding_mixture():
    dist = binomial_rounding_mixture(n=10.5, p=0.4)
    m, v = dist.mean(), dist.var()
    print(f"BinomRoundMix(n=10.5, p=0.4) mean: {m:.2f}, var: {v:.2f}")
    assert -1e-2 < dist.pmf(np.arange(20)).sum() - 1 < 1e-2
    sample = dist.rvs(size=100_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    np.testing.assert_allclose(counts / 100_000, dist.pmf(vals), atol=1e-2)
    # check the mean & variance agree with the pmf & sample
    assert -1e-1 < (dist.pmf(vals) * vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1


def test_com_poisson():
    dist = com_poisson(lam=10.5, nu=1.5)
    m, v = dist.mean(), dist.var()
    print(f"ComPoisson(mu=10.5, nu=1.5) mean: {m:.2f}, var: {v:.2f}")
    # TODO: PMF needs vectorizing
    # assert -1e-2 < dist.pmf(np.arange(20)).sum() - 1 < 1e-2
    # sample = dist.rvs(size=100_000)
    # assert -1e-1 < sample.mean() - m < 1e-1
    # assert -1e-1 < sample.var() - v < 1e-1


def test_delaporte():
    dist = delaporte(lam=10.5, a=1.5, b=0.5)
    m, v = dist.mean(), dist.var()
    m, v, s = dist.stats(moments="mvs")
    print(f"Delaporte(lam=10.5, a=1.5, b=0.5) mean: {m:.2f}, var: {v:.2f}, skew: {s:.2f}")
    # TODO: PMF needs vectorizing
    # assert -1e-2 < dist.pmf(np.arange(20)).sum() - 1 < 1e-2
    sample = dist.rvs(size=1_000_000)
    # check that the pmf and value_counts from the sample are close
    vals, counts = np.unique(sample, return_counts=True)
    # np.testing.assert_allclose(counts/1_000_000, dist.pmf(vals), atol=1e-3)
    # check the mean & variance agree with the pmf & sample
    # assert -1e-1 < (dist.pmf(vals)*vals).sum() - m < 1e-1
    assert -1e-1 < sample.mean() - m < 1e-1
    assert -1e-1 < sample.var() - v < 1e-1
    assert -1e-1 < scipy.stats.skew(sample) - s < 1e-1
