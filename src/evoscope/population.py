"""Weighted particle populations: the finite-support laws the toolbox works with.

A population is a probability measure with finitely many atoms, given by an
(n, d) array of points and n nonnegative weights summing to one.  Equal weights
are the usual case; unequal weights arise after selection.  The paper writes
such a law as mu (or its normalization mu-bar); here every Population is normalized.
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["Population", "broad_gaussian", "two_cluster"]


@dataclass(frozen=True)
class Population:
    points: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        points = np.atleast_2d(np.asarray(self.points, dtype=float))
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (len(points),):
            raise ValueError(f"weights shape {weights.shape} does not match {len(points)} points")
        if (weights < 0).any() or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12):
            raise ValueError("weights must be nonnegative and sum to one")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def uniform(cls, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return cls(points, np.full(len(points), 1/len(points)))

    @property
    def size(self):
        return len(self.points)

    @property
    def dimension(self):
        return self.points.shape[1]

    def expect(self, values):
        """Integral of a function given by its values at the atoms."""
        return float(np.sum(self.weights*np.asarray(values)))

    def mean(self):
        return self.weights@self.points

    def variance(self):
        """Var(mu) = int ||x - mean||^2 dmu, the spread observable of the paper."""
        centered = self.points-self.mean()
        return float(np.sum(self.weights*np.sum(centered*centered, axis=1)))

    def sample(self, size, rng):
        """Indices of `size` independent draws from the atoms."""
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.
        return np.searchsorted(cdf, rng.random(size), side="right")


def broad_gaussian(n, d, rng, mean=.6, scale=1.2):
    """The broad initial population of the study: n points from N(mean 1, scale^2 I)."""
    return Population.uniform(rng.normal(mean, scale, (n, d)))


def two_cluster(n, d, rng, center=1., scale=.1):
    """The clustered initial population of the study: each point near +center 1 or -center 1
    (the sign is shared across coordinates), with Gaussian spread `scale`."""
    centers = center*(2*rng.integers(0, 2, size=(n, 1))-1)
    return Population.uniform(centers+scale*rng.normal(size=(n, d)))
