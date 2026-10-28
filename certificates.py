"""One-sided IID certificates for paired excess classification error.

D = error(candidate)-error(reference) in {-1,0,1}. All methods test
H0: E[D] >= epsilon. Return valid p-value upper bounds (not exact p-values).
No claim of a new general statistical principle is made.
"""
import numpy as np
from scipy.special import betaln, gammaln, logsumexp
from scipy.optimize import minimize_scalar
from scipy.stats import beta, binom


def cp_upper(k, n, delta):
    return 1. if k == n else float(beta.ppf(1-delta, k+1, n-k))


def cp_lower(k, n, delta):
    return 0. if k == 0 else float(beta.ppf(delta, k, n-k+1))


def universal_log_e(d, eps, independent_probs=None, priors=None):
    """Dirichlet-mixture likelihood / constrained null maximum likelihood.

    Categories are benefit, tie, harm. Null maximizer lies on harm-benefit
    = eps when the empirical mean is below eps. Mixture weights and prior
    concentrations are fixed before any data, including tuning data.
    """
    counts = np.array([(d == -1).sum(), (d == 0).sum(), (d == 1).sum()])
    n = len(d)
    if np.mean(d) >= eps:
        return -np.inf
    def neg_ll(b):
        p = np.array([b, 1-2*b-eps, b+eps])
        return -np.sum(counts*np.log(np.maximum(p, 1e-300)))
    # Concavity gives a global upper bound from ANY interior tangent.
    # This guards against anticonservative numerical MLE underestimation.
    opt = minimize_scalar(neg_ll, bounds=(0., (1-eps)/2), method='bounded',
                          options={'xatol': 1e-14})
    b = opt.x
    slope = counts[0]/b - 2*counts[1]/(1-eps-2*b) + counts[2]/(b+eps)
    tangent_max = -opt.fun + max(slope*(0-b),slope*((1-eps)/2-b))
    null_ll = max(tangent_max,-neg_ll(0.),-neg_ll((1-eps)/2)) + 1e-8
    if priors is None:
        priors = [np.array([.5,.5,.5]), np.array([.1,5.,.1]), np.array([1.,10.,1.])]
    logs = [gammaln(a.sum())-gammaln(a.sum()+n)
            +np.sum(gammaln(a+counts)-gammaln(a)) for a in priors]
    numerator = (logsumexp(logs)-np.log(len(logs)) if independent_probs is None
                 else np.sum(counts*np.log(independent_probs)))
    return float(numerator-null_ll)


def log_e_betting(d, eps, adaptive=False):
    if adaptive:
        # Predictable plug-in stake, bounded away from bankruptcy. Past only.
        past_sum = np.r_[0., np.cumsum(d)[:-1]]
        past_sq = np.r_[0., np.cumsum(d*d)[:-1]]
        t = np.arange(len(d))
        mu = past_sum/(t+1)
        second = (past_sq+1)/(t+1)
        stake = np.clip((eps-mu)/(second+eps*eps), 0., .95/(1-eps))
        return float(np.log1p(stake*(eps-d)).sum())
    stakes = np.geomspace(.002, .95/(1-eps), 40)
    logs = np.log1p(stakes[:,None]*(eps-d[None,:])).sum(axis=1)
    return float(logsumexp(logs)-np.log(len(stakes)))


def certified(d, eps, delta, method):
    d = np.asarray(d, dtype=float)
    n = len(d)
    if not n:
        return False
    if method == 'ternary_universal':
        return universal_log_e(d, eps) >= np.log(1/delta)
    if method == 'mixture_betting':
        return log_e_betting(d, eps) >= np.log(1/delta)
    if method == 'predictable_betting':
        return log_e_betting(d, eps, True) >= np.log(1/delta)
    if method == 'hoeffding':
        return d.mean()+np.sqrt(2*np.log(1/delta)/n) <= eps
    if method == 'empirical_bernstein':
        if n < 2:
            return False
        u = d.mean()+np.sqrt(2*d.var(ddof=1)*np.log(2/delta)/n)
        u += 14*np.log(2/delta)/(3*(n-1))
        return u <= eps
    if method == 'split_cp':
        return cp_upper(int((d==1).sum()),n,delta/2)-cp_lower(int((d==-1).sum()),n,delta/2) <= eps
    if method == 'harm_only':
        return binom.cdf(int((d==1).sum()),n,eps) <= delta
    if method == 'naive':
        return d.mean() <= eps
    raise ValueError(method)


METHODS = ['ternary_universal','mixture_betting','predictable_betting',
           'empirical_bernstein','hoeffding','split_cp','harm_only','naive']


def choose(losses, eps, delta, method, correction='fixed'):
    """Columns ordered from conservative to aggressive BEFORE calibration.

    Fixed sequence stops at first failure; no marginal risk monotonicity
    assumption. Fallback is identically the reference, risk exactly zero.
    Bonferroni permits arbitrary selection from all certified candidates.
    """
    selected = -1
    for j in range(losses.shape[1]):
        level = delta if correction == 'fixed' else delta/losses.shape[1]
        if certified(losses[:,j], eps, level, method):
            selected = j
        elif correction == 'fixed':
            break
    return selected
