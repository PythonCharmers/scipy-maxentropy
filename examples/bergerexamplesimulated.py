#!/usr/bin/env python

""" Example use of the maximum entropy module fit a model using
    simulation:

    Machine translation example -- English to French -- from the paper 'A
    maximum entropy approach to natural language processing' by Berger et
    al., 1996.

    Consider the translation of the English word 'in' into French.  We
    notice in a corpus of parallel texts the following facts:

        (1)    p(dans) + p(en) + p(a) + p(au cours de) + p(pendant) = 1
        (2)    p(dans) + p(en) = 3/10
        (3)    p(dans) + p(a)  = 1/2

    This code finds the probability distribution with maximal entropy
    subject to these constraints.

    This problem is small enough to solve analytically, but this code
    shows the steps one would take to fit a model on a continuous or
    large discrete sample space.
"""

__author__ = "Ed Schofield"


import sys
import scipy_maxentropy as maxentropy
import scipy_maxentropy.maxentutils as utils


try:
    algorithm = sys.argv[1]
except IndexError:
    algorithm = "CG"
else:
    assert algorithm in ["CG", "BFGS", "LBFGSB", "Powell", "Nelder-Mead"]

a_grave = "\u00e0"

samplespace = ["dans", "en", a_grave, "au cours de", "pendant"]


def f0(x):
    return x in samplespace


def f1(x):
    return x == "dans" or x == "en"


def f2(x):
    return x == "dans" or x == a_grave


f = [f0, f1, f2]

model = maxentropy.BigModel()

# Now set the desired feature expectations
b = [1.0, 0.3, 0.5]

# Define a uniform instrumental distribution for sampling
samplefreq = {}
for e in samplespace:
    samplefreq[e] = 1

n = 10**4
m = 3

sampler = utils.dictsampler(samplefreq, size=n)


# Now create a generator of features of random points:
def sampleFgen(sampler, f, sparse_format="csc_matrix"):
    """
    A generator function that yields features of random points.

    Parameters
    ----------
        sampler: a generator that yields tuples (xs, logprobs)

        f: a list of feature functions to apply to the values x in xs

        sparse_format: either 'csc_matrix', 'csr_matrix' etc.
                       for constructing a scipy.sparse matrix of features

    Yields
    ------
        a tuple (F, logprobs), where:
            - F is a sparse feature matrix
            - logprobs is the same vector of log probs yielded by sampler
    """
    while True:
        xs, logprobs = next(sampler)
        F = maxentropy.sparsefeaturematrix(f, xs, sparse_format)
        yield F, logprobs


print("Generating an initial sample ...")
model.setsampleFgen(sampleFgen(sampler, f))

model.verbose = False

# Fit the model
model.avegtol = 1e-4
model.fit(b, algorithm=algorithm)

# Output the distribution
print()
print("Fitted model parameters are:\n" + str(model.params))
print()
smallmodel = maxentropy.Model(f, samplespace)
smallmodel.setparams(model.params)
print("Fitted distribution is:")
p = smallmodel.probdist()
for j in range(len(smallmodel.samplespace)):
    x = smallmodel.samplespace[j]
    print(f"    x = {x + ':':15s} p(x) = {p[j]:.3f}")

# Now show how well the constraints are satisfied:
print()
print("Desired constraints:")
print("    sum(p(x))           = 1.0")
print("    p['dans'] + p['en'] = 0.3")
print("    p['dans'] + p['à']  = 0.5")
print()
print("Actual expectations under the fitted model:")
print(f"    sum(p(x))           = {p.sum():.3f}")
print(f"    p['dans'] + p['en'] = {p[0] + p[1]:.3f}")
print(f"    p['dans'] + p['à']  = {p[0] + p[2]:.3f}")

print(
    "\nEstimated error in constraint satisfaction (should be close to 0):\n"
    + str(abs(model.expectations() - b))
)
print(
    "\nTrue error in constraint satisfaction:\n"
    + str(abs(smallmodel.expectations() - b))
)
print()
print("The true error will be closer to 0 for larger samples n.\n")
