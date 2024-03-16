# scipy-maxentropy: maximum entropy models

This is the former `scipy.maxentropy` package that was available in SciPy up to
version 0.10.1. It was under-maintained and later removed in SciPy 0.11. It is
now available as this separate package for backward compatibility.

For new projects, consider the
[maxentropy](https://github.com/PythonCharmers/maxentropy) package instead,
which offers a more modern scikit-learn compatible API.

## Purpose

This package fits "exponential family" models, including models of maximum
entropy and minimum KL divergence to other models, subject to linear constraints
on the expectations of arbitrary feature statistics. Applications include
language models for natural language processing and understanding, machine
translation, etc., environmental species modelling, image reconstruction, and
others.

## Quickstart

Here is a quick usage example based on the trivial machine translation example
from the paper 'A maximum entropy approach to natural language processing' by
Berger et al., Computational Linguistics, 1996.

Consider the translation of the English word 'in' into French. Assume we notice
in a corpus of parallel texts the following facts:

    (1)    p(dans) + p(en) + p(à) + p(au cours de) + p(pendant) = 1
    (2)    p(dans) + p(en) = 3/10
    (3)    p(dans) + p(à)  = 1/2

This code finds the probability distribution with maximal entropy subject to
these constraints.

```python
from scipy_maxentropy import Model    # previously scipy.maxentropy

samplespace = ['dans', 'en', 'à', 'au cours de', 'pendant']

def f0(x):
    return x in samplespace

def f1(x):
    return x=='dans' or x=='en'

def f2(x):
    return x=='dans' or x=='à'

f = [f0, f1, f2]

model = Model(f, samplespace)

# Now set the desired feature expectations
b = [1.0, 0.3, 0.5]

model.verbose = False    # set to True to show optimization progress

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
```

## Models available

These model classes are available:
- `scipy_maxentropy.Model`: for models on discrete, enumerable sample spaces
- `scipy_maxentropy.ConditionalModel`: for conditional models on discrete, enumerable sample spaces
- `scipy_maxentropy.BigModel`: for models on sample spaces that are either continuous (and
perhaps high-dimensional) or discrete but too large to enumerate, like all possible
sentences in a natural language. This model uses conditional Monte Carlo methods
(primarily importance sampling).

## Background

This package fits probabilistic models of the following exponential form:

$$
   p(x) = p_0(x) \exp(\theta^T f(x)) / Z(\theta; p_0)
$$

with a real parameter vector $\theta$ of the same length $n$ as the feature
statistics $f(x) = \left(f_1(x), ..., f_n(x)\right)$.

This is the "closest" model (in the sense of minimizing KL divergence or
"relative entropy") to the prior model $p_0$ subject to the following additional
constraints on the expectations of the features:

```
    E f_1(X) = b_1
    ...
    E f_n(X) = b_n
```

for some constants $b_i$, such as statistics estimated from a dataset.

In the special case where $p_0$ is the uniform distribution, this is the
"flattest" model subject to the constraints, in the sense of having **maximum
entropy**.

For more background, see, for example, Cover and Thomas (1991), *Elements of
Information Theory*.


