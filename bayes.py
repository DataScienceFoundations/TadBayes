"""This file contains code useful in Bayesian statistics
"""

"""This file contains class definitions for:
Cdf: discrete cumulative distribution function
Pdf: continuous probability density function
Pmf: probability mass function (probabilities for given values)

Also:
GPdf: Gaussian PDF
Beta: Beta Distribution
"""

import numpy as np
import pandas as pd
import random
import scipy.stats as sp_stats


def odds_for_prob(prob):
    """Returns odds for a given decimal probability (for)"""
    if prob == 1:
        return float('inf')
    return prob / (1 - prob)


def prob_for_odds_decimal(o):
    """Returns probability for odds
    
    o=2 represents 2:1 odds
    """
    return o / (o + 1)


def prob_for_odds_in_favor(a, b=1):
    """Returns probability for odds in favor
    
    a=3, b=2 represents 3:2 odds in favor
    """
    return a / (a + b)


def prob_for_odds_against(a, b=1):
    """Returns probability for odds against
    
    a=3, b=2 represents 3:2 odds against
    """
    return prob_for_odds_in_favor(b, a)


class DistBase(pd.Series):
    """The base class for all distributions
    
    Allows uninitilized index to be incremented.
    """
    
    def inc(self, i, value=1):
        self.at[i] = self.get(i, 0.0) + value


class Pmf(DistBase):
    """The base class for all Pmfs"""
    
    def __add__(self, other):
        """Adds two PMFs
        
        If other is another PMF, calls add_pmf.
        If other is not PMF, adds by index.
        To add two PMFs by index, use add().
        This function automatically uses a fill_value
        of 0.0 if needed.
        """
        if type(self) == type(other):
            return self.add_pmf(other)
        else:
            return self.add(other, fill_value=0.0)
        
    def __sub__(self, other):
        """Subtracts other PMF from this PMF
        
        If other is another PMF, calls sub_pmf.
        If other is not PMF, subtracts by index.
        To subtract two PMFs by index, use sub().
        This function automatically uses a fill_value
        of 0.0 if needed.
        """
        if type(self) == type(other):
            return self.sub_pmf(other)
        else:
            return self.sub(other, fill_value=0.0)
        
    def add_pmf(self, other):
        """Adds two PMFs together"""
        pmf = Pmf()
        for si, sv in self.items():
            for oi, ov in other.items():
                pmf.inc(si+oi, sv*ov)
        return pmf
    
    def sub_pmf(self, other):
        """Subtract other from this PMF"""
        pmf = Pmf()
        for si, sv in self.items():
            for oi, ov in other.items():
                pmf.inc(si-oi, sv*ov)
        return pmf
        
    def pmf_mean(self):
        """Gets the mean of this PMF"""
        return sum(self.index * self.values)
    
    def pmf_sample(self, n=1, replace=True):
        """Gets a sample from the PMF
        
        The chance for sampling a certain value is
        weighted according to the PMFs current posterior.
        """
        return self.sample(n=n, weights=self.values, replace=replace).index
    
    def pmf_copy(self):
        """Gets a copy of this PMF"""
        return Pmf(self.copy())
        
    def normalize(self):
        """Normalizes this PMF between 0 and 1"""
        factor = 1 / self.sum()
        self *= factor
        
    def update(self, data):
        """Updates this PMF with the given data"""
        for hypo in self.index:
            self.at[hypo] *= self.like(data, hypo)
        self.normalize()
        
    def update_set(self, *data):
        """Updates this PMF with the given set of data
        
        Only normalizes once, as opposed to calling
        update, which normalizes after each update.
        """
        for datum in data:
            for hypo in self.index:
                self.at[hypo] *= self.like(datum, hypo)
        self.normalize()
    
    def like(self, data, hypo):
        """The likelihood function for this PMF
        
        Must be overridden based on current PMF
        implementation.
        """
        raise UnimplementedMethodException()
    
    def max_likelihood(self):
        """Gets the value with the maximum likelihood of occurring"""
        return self[(self == self.max())].index[0]
    
    def credible_interval(self, percentage=90):
        """Gets the credible interval of the PMF
        
        The interval most likely to occur based on
        the PMFs current posterior.
        """
        cdf = self.to_cdf()
        return cdf.credible_interval(percentage)
    
    def percentile(self, percentage):
        """Gets the value associated with a percentage from this PMF"""
        cdf = self.to_cdf()
        return cdf.percentile(percentage)
       
    def pmf_credible_interval(self, percentage=90):
        """Gets the credible interval of the PMF
        
        The interval most likely to occur based on
        the PMFs current posterior.
        """
        prob = (100 - percentage) / 2
        return self.pmf_percentile(prob), self.pmf_percentile(100-prob)
    
    def pmf_percentile(self, percentage):
        """Gets the value associated with a percentage from this PMF"""
        i = 0
        total = 0
        p = percentage / 100.0
        for v in self.values:
            total += v
            if total >= p:
                return self.index[i]
            i += 1
    
    def prob(self, p, default=0.0):
        """Gets the probability of a value occurring
        
        As opposed to using 'at', this avoids indexing
        non-existent indexes with the 'get' function.
        """
        return self.get(p, default)
    
    def probs(self, seq, default=0.0):
        """Gets the probabilities of a value occurring
        
        As opposed to using 'loc', this avoids indexing
        non-existent indexes with the 'get' function.
        """
        return pd.Series([self.get(p, default) for p in seq])
    
    def prob_greater(self, other):
        """Gets the probability this PMF > other PMF"""
        total = 0.0
        for si, sv in self.items():
            for oi, ov in other.items():
                if si > oi:
                    total += sv * ov
        return total
    
    def prob_less(self, other):
        """Gets the probability this PMF < other PMF"""
        total = 0.0
        for si, sv in self.items():
            for oi, ov in other.items():
                if si < oi:
                    total += sv * ov
        return total
    
    def prob_equal(self, other):
        """Gets the probability this PMF == other PMF"""
        total = 0.0
        for si, sv in self.items():
            for oi, ov in other.items():
                if si == oi:
                    total += sv * ov
        return total
    
    def prob_sample_greater(self, x):
        """Gets the probability a sample for this PMF > a value x"""
        return self[(self > x)].sum()
    
    def prob_sample_less(self, x):
        """Gets the probability a sample for this PMF < a value x"""
        return self[(self < x)].sum()

    def to_cdf(self):
        """Returns a CDF representation of this PMF"""
        return Cdf(self.cumsum())
    
    def to_max_cdf(self, k):
        """Returns a maximum distribution CDF for this PMF
        
        Represents the distribution for the probability of
        getting a value as a maximum after k samples.
        """
        cdf = self.to_cdf()
        return cdf.to_max_cdf(k)
    
    
class Cdf(DistBase):
    """Class representing CDFs"""
    
    def normalize(self):
        """Normalizes this CDF between 0 and 1"""
        factor = 1 / self.iat[-1]
        self *= factor
        
    def percentile(self, p):
        """Gets the value associated with a percentage from this CDF"""
        if p == 0: return self.index[0]
        if p == 1: return self.index[-1]
        return self.index[self.searchsorted(p / 100.0)[0]]
    
    def prob(self, x):
        """Gets the probability of a value based on this CDF"""
        if x < self.index[0]: return 0.0
        if x > self.index[-1]: return self.max()
        return self.at[self.index[self.index.searchsorted(x)-1]]
    
    def credible_interval(self, percentage=90):
        """Gets the credible interval of the CDF
        
        The interval most likely to occur based on
        the CDFs current distribution.
        """
        prob = (1 - percentage / 100.0) / 2
        return tuple(self.index[self.searchsorted([prob, 1-prob])])
    
    def to_max_cdf(self, k):
        """Returns a maximum distribution CDF for this CDF
        
        Represents the distribution for the probability of
        getting a value as a maximum after k samples.
        """
        return Cdf([p**k for p in self.values], index=self.index)
    
    def to_pmf(self):
        """Returns a PMF representation of this CDF"""
        pmf = Pmf()
        prev = 0.0
        for i, v in self.items():
            pmf.inc(i, v - prev)
            prev = v
        return pmf


class BasePdf:
    """Base class for all PDFs"""
    
    def density(self, x):
        """Returns the density of of PDF(x)
        
        Must be implemented by inheriting
        PDF implementations.
        """
        raise UnimplementedMethodException()
        
    def to_pmf(self, xs):
        """Returns a PMF representation of this PDF"""
        pmf = Pmf()
        for x in xs:
            pmf.at[x] = self.density(x)
        pmf.normalize()
        return pmf


class GPdf(BasePdf):
    """Class representing Gaussian PDF"""
    
    def __init__(self, mu, sigma):
        """Constructs Guassian PDF.
        
        mu: mean
        sigma: standard deviation
        """
        self.mu = mu
        self.sigma = sigma
        
    def density(self, x):
        """Returns the density of GPDF(x)"""
        return sp_stats.norm.pdf(x, self.mu, self.sigma)
    
    def gpdf_eval(self, x):
        """Returns the evaluation of this GPDF"""
        return sp_stats.norm.pdf(x, self.mu, self.sigma)
    
    
class Pdf(BasePdf):
    """Class representing PDFs"""
    
    def __init__(self, sample):
        """Constructs basic PDF based on sample"""
        self.kde = sp_stats.gaussian_kde(sample)
        
    def density(self, x):
        """Returns teh density of PDF(x)"""
        return self.kde.evaluate(x)
    
    def to_pmf(self, xs):
        """Returns a PMF representation of this PDF"""
        values = self.kde.evaluate(xs)
        return Pmf(values, index=xs)
    
    
class Beta:
    """Class representing Beta Distribution"""
    
    def __init__(self, alpha=1.0, beta=1.0):
        """Constructs a Beta distribution"""
        self.alpha = alpha
        self.beta = beta
        
    def update(self, data):
        """Updates this Beta distribution with the given data"""
        alpha, beta = data
        self.alpha = alpha
        self.beta = beta
    
    def mean(self):
        """Gets the mean of this beta distribution"""
        return self.alpha / (self.alpha + self.beta)
    
    def random(self):
        """Gets a random variate from this beta distribution"""
        return random.betavariate(self.alpha, self.beta)
    
    def sample(self, n):
        """Gets a random sample from this beta distribution"""
        return np.random.beta(self.alpha, self.beta, n)
    
    def eval_pdf(self, x):
        """Returns the evaluation of PDF(x) based on this beta distribution"""
        return x**(self.alpha - 1) * (1 - x)**(self.beta - 1)
    
    def to_pmf(self, steps=101):
        """Returns a PMF representation of this beta distribution"""
        if self.alpha < 1 or self.beta < 1:
            cdf = self.to_cdf()
            return cdf.to_pmf()
        index = [i / (steps - 1) for i in range(steps)]
        values = [self.eval_pdf(i) for i in index]
        return Pmf(values, index=index)


#######################################
# Functions for classes defined above #
#######################################


def get_init_pmf(hypos, normalize=True):
    """gets a pmf with a uniform prior"""
    pmf = Pmf(1.0, index=hypos)
    if normalize: pmf.normalize()
    return pmf


def get_init_pow_law_prior_pmf(hypos, alpha=1.0, normalize=True):
    """gets a pmf with a power law prior"""
    pmf = Pmf([hypo**(-alpha) for hypo in hypos], index=hypos)
    if normalize: pmf.normalize()
    return pmf


def get_pmf_from_seq(seq, normalize=True):
    """gets a pmf from a sequence of data
    
    The data, unlike with 'get_init_pmf',
    can contain duplicate values.
    """
    pmf = Pmf()
    pmf_seq = seq[:]
    pmf_seq.sort()
    for v in pmf_seq:
        pmf.inc(v)
    if normalize: pmf.normalize()
    return pmf


def get_pmf_from_dict(d, normalize=True):
    """gets a pmf from a dictionary
    
    The dictionary must be formatted
    as a PMF.
    """
    pmf = Pmf(d)
    if normalize: pmf.normalize()
    return pmf


def get_meta_pmf(pmfs, probs, normalize=True):
    """Gets a meta-PMF.
    
    A meta-PMF is a PMF representing the
    probabilities of other PMFs.
    """
    pmf = Pmf(probs, index=pd.Index(pmfs, dtype='object'))
    if normalize: pmf.normalize()
    return pmf


def get_mix_pmf(meta_pmf, normalize=True):
    """Gets a mixture distribution
    
    A mixture is a representation of the probability
    of an event occurring based on multiple PMFs.
    Each of the PMFs may or may not have an event with
    an associated probability of that even occurring.
    """
    mix = Pmf()
    for pmf, outer_v in meta_pmf.items():
        for i, inner_v in pmf.items():
            mix.inc(i, outer_v*inner_v)
    if normalize: pmf.normalize()
    return mix


def get_random_sum(pmfs):
    """Gets the sum of a random sample from multiple distributions"""
    return sum(pmf.sample().index[0] for pmf in pmfs)


def get_sample_sum_pmf(pmfs, n, normalize=True):
    """Gets a PMF representing the sum of 'n' random samples of a distribution"""
    return get_pmf_from_seq([get_random_sum(pmfs) for _ in range(n)],
                            normalize=normalize)


def get_sample_max_pmf(pmf, samples, iters=100, normalize=True):
    """Gets a PMF representing the distribution of the maximum of a number of samples"""
    sample_max_pmf = Pmf()
    for _ in range(iters):
        sample_max_pmf.inc(max(pmf.pmf_sample(n=samples)))
    sample_max_pmf.sort_index(inplace=True)
    sample_max_pmf = Pmf(sample_max_pmf.reindex(pmf.index, fill_value=0.0))
    if normalize: pmf.normalize()
    return sample_max_pmf


def max_pmf(pmf1, pmf2, normalize=False):
    """Gets a PMF representing the maximum distribution of two PMFs"""
    pmf = Pmf()
    for i1, v1 in pmf1.items():
        for i2, v2 in pmf2.items():
            pmf.inc(max(i1,i2), v1*v2)
    if normalize: pmf.normalize()
    return pmf


def get_init_cdf(hypos):
    """Gets a CDF representation of a uniform-prior PMF"""
    pmf = get_init_pmf(hypos)
    return pmf.to_cdf()


def get_cdf_from_seq(seq):
    """Gets a CDF representation of a PMF initialized by a sequence of data
    
    The data, unlike with 'get_init_cdf',
    can contain duplicate values.
    """
    pmf = get_pmf_from_seq(seq)
    return pmf.to_cdf()


def get_cdf_from_dict(d):
    """Gets a CDF representation of a PMF initialized by a dictionary
    
    The dictionary must already be formatted
    as a PMF.
    """
    pmf = get_pmf_from_dict(d)
    return pmf.to_cdf()


def get_cdf_from_cdf_dict(d):
    """Gets a CDF from a dictionary in CDF form"""
    cdf = Cdf(d)
    cdf.normalize()
    return cdf


def get_index(seq, dtype):
    """Gets a pd.Index object of specific type for a sequence of data"""
    return pd.Index(seq, dtype=dtype)

