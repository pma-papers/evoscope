"""Test problems with exact derivatives and Gaussian expectations.

The ten problem--dimension instances of the paper's numerical study.  Every problem is smooth with minimum value zero and
provides the four quantities the toolbox needs: the objective, its gradient and
Laplacian (for generator actions), and the Gaussian expectation
E f(m + sqrt(v) Z) with Z standard normal (for exact finite steps of Gaussian
mutation).  All functions act on the last axis of their input.

Any object with ``value``, ``gradient`` and ``laplacian`` methods (and optionally
``gaussian_mean``) can be used wherever a :class:`Problem` is accepted; see
:class:`evoscope.operators.Observable` for user-defined test functions.
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Problem", "panel", "get"]

_FORMULAS = {
    "separable_pl": "sum_j (x_j^2 + 1.05 sin^2 x_j)",
    "rotated_anisotropic_pl": "sum_j a_j (z_j^2 + 1.05 sin^2 z_j), z = Q^T x, a_j linear from 1 to 2",
    "quartic": "sum_j (x_j^2 - 1)^2",
    "rastrigin": "sum_j [x_j^2 + 10 (1 - cos 2 pi x_j)]",
    "rosenbrock": "sum_{j<d} [(z_{j+1} - z_j^2)^2 + 0.01 (1 - z_j)^2], z = Q^T x",
    "wells": "(x_1^2 - 1)^2 + 0.2 (x_1 - 1)^2 + 0.5 sum_{j>1} x_j^2",
    "periodic": "1 - cos x_1",
}
_NAMES = {
    "separable_pl": "Separable nonconvex PL", "rotated_anisotropic_pl": "Rotated anisotropic PL",
    "quartic": "Multimodal quartic", "rastrigin": "Rastrigin", "rosenbrock": "Rotated scaled Rosenbrock",
    "wells": "Unequal wells", "periodic": "Periodic",
}


@dataclass(frozen=True)
class Problem:
    """A smooth test objective on R^d with f_* = 0."""
    family: str
    dimension: int
    rotation: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    target: np.ndarray = field(repr=False)
    f_star: float = 0.0

    @property
    def id(self):
        return f"{self.family}_d{self.dimension}"

    @property
    def name(self):
        return f"{_NAMES[self.family]} {self.dimension}D"

    @property
    def formula(self):
        return _FORMULAS[self.family]

    # ------------------------------------------------------------------ values
    def value(self, x):
        x = np.asarray(x, dtype=float)
        family = self.family
        with np.errstate(over="ignore", invalid="ignore"):
            if family in ("separable_pl", "rotated_anisotropic_pl"):
                z = x if family == "separable_pl" else x@self.rotation
                return np.sum(self.weights*(z*z+1.05*np.sin(z)**2), axis=-1)
            if family == "quartic":
                return np.sum((x*x-1)**2, axis=-1)
            if family == "rastrigin":
                return np.sum(x*x+10*(1-np.cos(2*np.pi*x)), axis=-1)
            if family == "rosenbrock":
                z = x@self.rotation
                return np.sum((z[..., 1:]-z[..., :-1]**2)**2+.01*(1-z[..., :-1])**2, axis=-1)
            if family == "wells":
                return (x[..., 0]**2-1)**2+.2*(x[..., 0]-1)**2+.5*np.sum(x[..., 1:]**2, axis=-1)
            if family == "periodic":
                return 1-np.cos(x[..., 0])
        raise ValueError(family)

    def gradient(self, x):
        return self._derivatives(np.asarray(x, dtype=float))[0]

    def laplacian(self, x):
        return self._derivatives(np.asarray(x, dtype=float))[1]

    def _derivatives(self, x):
        family = self.family
        if family in ("separable_pl", "rotated_anisotropic_pl"):
            Q, a = self.rotation, self.weights
            z = x@Q
            return (a*(2*z+1.05*np.sin(2*z)))@Q.T, np.sum(a*(2+2.1*np.cos(2*z)), axis=-1)
        if family == "quartic":
            return 4*x*(x*x-1), np.sum(12*x*x-4, axis=-1)
        if family == "rastrigin":
            return (2*x+20*np.pi*np.sin(2*np.pi*x),
                    np.sum(2+40*np.pi*np.pi*np.cos(2*np.pi*x), axis=-1))
        if family == "rosenbrock":
            Q = self.rotation
            z = x@Q
            difference = z[..., 1:]-z[..., :-1]**2
            dz = np.zeros_like(z)
            dz[..., :-1] += -4*z[..., :-1]*difference+.02*(z[..., :-1]-1)
            dz[..., 1:] += 2*difference
            return dz@Q.T, np.sum(12*z[..., :-1]**2-4*z[..., 1:]+2.02, axis=-1)
        if family == "wells":
            gradient = x.copy()
            gradient[..., 0] = 4*x[..., 0]*(x[..., 0]**2-1)+.4*(x[..., 0]-1)
            return gradient, 12*x[..., 0]**2-3.6+(self.dimension-1)
        if family == "periodic":
            return np.sin(x), np.cos(x[..., 0])
        raise ValueError(family)

    def gaussian_mean(self, mean, variance):
        """E f(mean + sqrt(variance) Z), Z ~ N(0, I_d), in closed form (variance a scalar)."""
        if variance < 0:
            raise ValueError("variance must be nonnegative")
        mean, v, family = np.asarray(mean, dtype=float), float(variance), self.family
        if family in ("separable_pl", "rotated_anisotropic_pl"):
            z = mean@self.rotation
            return np.sum(self.weights*(z*z+v+.525*(1-np.exp(-2*v)*np.cos(2*z))), axis=-1)
        if family == "quartic":
            return np.sum((mean*mean-1)**2+(6*mean*mean-2)*v+3*v*v, axis=-1)
        if family == "rastrigin":
            return np.sum(mean*mean+v+10*(1-np.exp(-2*np.pi*np.pi*v)*np.cos(2*np.pi*mean)), axis=-1)
        if family == "rosenbrock":
            z = mean@self.rotation
            previous, following = z[..., :-1], z[..., 1:]
            return np.sum((following-previous*previous)**2+(1+6*previous*previous-2*following)*v+3*v*v
                          +.01*((1-previous)**2+v), axis=-1)
        if family == "wells":
            x = mean[..., 0]
            return ((x*x-1)**2+(6*x*x-2)*v+3*v*v+.2*((x-1)**2+v)
                    +.5*np.sum(mean[..., 1:]**2+v, axis=-1))
        if family == "periodic":
            return 1-np.exp(-v/2)*np.cos(mean[..., 0])
        raise ValueError(family)

    # ------------------------------------------------------------ minimizers
    def distance2_to_minimizers(self, x):
        """Squared distance to the nearest global minimizer (all sign vectors for the
        quartic, the lattice 2 pi Z for the periodic problem)."""
        x = np.asarray(x, dtype=float)
        if self.family == "quartic":
            return np.sum((np.abs(x)-1)**2, axis=-1)
        if self.family == "periodic":
            return np.sum((x-2*np.pi*np.rint(x/(2*np.pi)))**2, axis=-1)
        return np.sum((x-self.target)**2, axis=-1)


def _make(family, d):
    seed = 260930+d if family == "rotated_anisotropic_pl" else 270929 if family == "rosenbrock" else None
    Q = np.eye(d) if seed is None else np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))[0]
    weights = np.linspace(1., 2., d) if family == "rotated_anisotropic_pl" else np.ones(d)
    target = np.zeros(d)
    if family == "quartic":
        target = np.ones(d)
    elif family == "rosenbrock":
        target = np.ones(d)@Q.T
    elif family == "wells":
        target[0] = 1.
    for array in (Q, weights, target):
        array.setflags(write=False)
    return Problem(family, d, Q, weights, target)


def panel():
    """The ten instances of the paper's numerical study, keyed by id (e.g. ``"quartic_d4"``)."""
    rows = [(f, d) for f in ("separable_pl", "rotated_anisotropic_pl", "quartic") for d in (4, 8)]
    rows += [("rastrigin", 4), ("rosenbrock", 4), ("wells", 4), ("periodic", 1)]
    return {p.id: p for p in (_make(f, d) for f, d in rows)}


def get(problem_id):
    """One instance of the panel by id."""
    return panel()[problem_id]
