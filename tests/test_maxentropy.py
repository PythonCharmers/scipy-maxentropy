#!/usr/bin/env python

"""
Test functions for maximum entropy module.
"""

from numpy import arange, log, exp, ones, isclose
from scipy_maxentropy import logsumexp


def test_logsumexp(self):
    """Test whether logsumexp() function correctly handles large
    inputs.
    """

    a = arange(200)
    desired = log(sum(exp(a)))
    assert isclose(logsumexp(a), desired)

    # Now test with large numbers
    b = [1000, 1000]
    desired = 1000.0 + log(2.0)
    assert isclose(logsumexp(b), desired)

    n = 1000
    b = ones(n) * 10000
    desired = 10000.0 + log(n)
    assert isclose(logsumexp(b), desired)
