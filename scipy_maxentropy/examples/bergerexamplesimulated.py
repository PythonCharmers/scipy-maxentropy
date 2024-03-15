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

__author__  =  'Ed Schofield'
__version__ =  '2.1'


import sys
import scipy_maxentropy as maxentropy
import scipy_maxentropy.maxentutils as utils


try:
    algorithm = sys.argv[1]
except IndexError:
    algorithm = 'CG'
else:
    assert algorithm in ['CG', 'BFGS', 'LBFGSB', 'Powell', 'Nelder-Mead']

a_grave = u'\u00e0'

samplespace = ['dans', 'en', a_grave, 'au cours de', 'pendant']

def f0(x):
    return x in samplespace

def f1(x):
    return x == 'dans' or x == 'en'

def f2(x):
    return x == 'dans' or x == a_grave

f = [f0, f1, f2]

model = maxentropy.BigModel()

# Now set the desired feature expectations
K = [1.0, 0.3, 0.5]

# Define a uniform instrumental distribution for sampling
samplefreq = {}
for e in samplespace:
    samplefreq[e] = 1

n = 10**4
m = 3

sampler = utils.dictsampler(samplefreq, size=n)

# Now create a generator of features of random points:
def sampleFgen(sampler, f, sparse_format='csc_matrix'):
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

model.verbose = True

# Fit the model
model.avegtol = 1e-4
model.fit(K, algorithm=algorithm)

# Output the true distribution
print("\nFitted model parameters are:\n" + str(model.params))
smallmodel = maxentropy.model(f, samplespace)
smallmodel.setparams(model.params)
print("\nFitted distribution is:")
p = smallmodel.probdist()
for j in range(len(smallmodel.samplespace)):
    x = smallmodel.samplespace[j]
    print(("\tx = %-15s" %(x + ":",) + " p(x) = "+str(p[j])))


# Now show how well the constraints are satisfied:
print()
print("Desired constraints:")
print("\tp['dans'] + p['en'] = 0.3")
print("\tp['dans'] + p['" + a_grave + "']  = 0.5")
print()
print("Actual expectations under the fitted model:")
print("\tp['dans'] + p['en'] =", p[0] + p[1])
print("\tp['dans'] + p['" + a_grave + "']  = " + \
        str(p[0]+p[2]))

print("\nEstimated error in constraint satisfaction (should be close to 0):\n" \
        + str(abs(model.expectations() - K)))
print("\nTrue error in constraint satisfaction (should be close to 0):\n" + \
        str(abs(smallmodel.expectations() - K)))
