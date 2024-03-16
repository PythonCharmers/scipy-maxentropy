#!/usr/bin/env python

""" Example use of the maximum entropy module:

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
"""

from scipy_maxentropy import Model  # previously scipy.maxentropy

samplespace = ["dans", "en", "à", "au cours de", "pendant"]


def f0(x):
    return x in samplespace


def f1(x):
    return x == "dans" or x == "en"


def f2(x):
    return x == "dans" or x == "à"


f = [f0, f1, f2]

model = Model(f, samplespace)

# Now set the desired feature expectations
b = [1.0, 0.3, 0.5]

model.verbose = False  # set to True to show optimization progress

# Fit the model
model.fit(b)

# Output the distribution
print()
print("Fitted model parameters are:\n" + str(model.params))
print()
print("Fitted distribution is:")
p = model.probdist()
for j in range(len(model.samplespace)):
    x = model.samplespace[j]
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
