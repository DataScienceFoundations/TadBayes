"""Optimal bid calculator from Think Bayes.

The code from Think Bayes has been changed to
match my implementation of the classes used
throughout Think Bayes.
"""

import sys
sys.path.append('../')

import bayes as b
import pandas as pd
import numpy as np


class Player:
    
    def __init__(self, prices, bids, diffs, xs):
        self.pdf_price = b.Pdf(prices)
        self.cdf_diff = b.get_cdf_from_seq(diffs)
        self.pdf_error = b.GPdf(0, np.std(diffs))
        self.xs = xs
        
    def error_density(self, error):
        return self.pdf_error.density(error)
    
    def make_beliefs(self, guess):
        pmf = self.pmf_price()
        self.prior = Price(pmf, self)
        self.posterior = Price(self.prior.copy(), self)
        self.posterior.update(guess)
    
    def pmf_price(self):
        return self.pdf_price.to_pmf(self.xs)
    
    def prob_overbid(self):
        return self.cdf_diff.prob(-1)
    
    def prob_worse_than(self, diff):
        return 1 - self.cdf_diff.prob(diff)
    
    def optimal_bid(self, guess, opp):
        self.make_beliefs(guess)
        calc = GainCalculator(self, opp)
        bids, gains = calc.expected_gains(self.xs)
        gain, bid = max(zip(gains, bids))
        self.bid_gain = pd.Series(dict(zip(bids, gains)))
        return bid, gain


class Price(b.Pmf):
    
    def __init__(self, pmf, player):
        b.Pmf.__init__(self, pmf)
        self.player = player
        
    def like(self, data, hypo):
        price = hypo
        guess = data
        error = price - guess
        return self.player.error_density(error)


class GainCalculator:
    
    def __init__(self, player, opp):
        self.player = player
        self.opp = opp
        
    def expected_gains(self, xs):
        gains = [self.expected_gain(bid) for bid in xs]
        return xs, gains
    
    def expected_gain(self, bid):
        pmf = self.player.posterior
        total = 0
        for price, prob in sorted(pmf.items()):
            gain = self.gain(bid, price)
            total += prob * gain
        return total
    
    def gain(self, bid, price):
        if bid > price:
            return 0
        diff = price - bid
        prob = self.prob_win(diff)
        if diff <= 250:
            return 2 * price * prob
        else:
            return price * prob
        
    def prob_win(self, diff):
        return (self.opp.prob_overbid() + self.opp.prob_worse_than(diff))
    
