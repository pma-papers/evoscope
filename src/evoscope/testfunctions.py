"""Test functions: the bounded smooth functions with which the conditions are probed.

The theory measures an operator through its effect on averages of bounded smooth test
functions (the space C_b^3 of the paper).  A numerical check cannot try every such
function, so it uses a battery: Gaussian bumps and cosine waves placed and scaled to the
populations being examined.  Each function knows its value, gradient, Laplacian, and an
upper bound on its bounded-Lipschitz norm, which the distance computations need.

``truncated(observable, radius)`` multiplies an unbounded function (an objective or a
Lyapunov function) by a smooth cutoff, the construction behind cutoff-admissible tests.
"""
import numpy as np

__all__ = ["TestFunction", "gaussian_bump", "cosine_wave", "battery", "truncated"]


class TestFunction:
    """A bounded smooth function with value, gradient, Laplacian and a bound on ||phi||_BL."""

    def __init__(self, value, gradient, laplacian, bl_norm, name, gaussian_mean=None):
        self.value, self.gradient, self.laplacian = value, gradient, laplacian
        self.bl_norm, self.name = float(bl_norm), name
        if gaussian_mean is not None:
            self.gaussian_mean = gaussian_mean

    def __repr__(self):
        return f"TestFunction({self.name})"


def gaussian_bump(center, scale):
    """phi(x) = exp(-||x - center||^2 / (2 scale^2)); sup = 1, Lipschitz constant 1/(scale sqrt(e))."""
    center, s2 = np.asarray(center, dtype=float), float(scale)**2

    def value(x):
        return np.exp(-np.sum((np.asarray(x)-center)**2, axis=-1)/(2*s2))

    def gradient(x):
        x = np.asarray(x)
        return -(x-center)/s2*value(x)[..., None]

    def laplacian(x):
        x = np.asarray(x)
        r2 = np.sum((x-center)**2, axis=-1)
        return (r2/s2**2-x.shape[-1]/s2)*value(x)

    def gaussian_mean(mean, variance):
        mean = np.asarray(mean)
        total = s2+variance
        return (s2/total)**(mean.shape[-1]/2)*np.exp(-np.sum((mean-center)**2, axis=-1)/(2*total))

    return TestFunction(value, gradient, laplacian, 1+1/(np.sqrt(s2)*np.sqrt(np.e)), f"bump(scale={scale:.3g})",
                        gaussian_mean)


def cosine_wave(frequency, phase):
    """phi(x) = cos(frequency . x + phase); sup = 1, Lipschitz constant ||frequency||."""
    w, b = np.asarray(frequency, dtype=float), float(phase)

    def value(x):
        return np.cos(np.asarray(x)@w+b)

    def gradient(x):
        return -np.sin(np.asarray(x)@w+b)[..., None]*w

    def laplacian(x):
        return -(w@w)*np.cos(np.asarray(x)@w+b)

    def gaussian_mean(mean, variance):
        return np.exp(-variance*(w@w)/2)*np.cos(np.asarray(mean)@w+b)

    return TestFunction(value, gradient, laplacian, 1+np.linalg.norm(w), f"cosine(|w|={np.linalg.norm(w):.3g})",
                        gaussian_mean)


def battery(populations, size=12, rng=None):
    """A battery of test functions adapted to the location and spread of the populations:
    half Gaussian bumps centred at atoms with scales comparable to the spread, half cosine
    waves with wavelengths comparable to the spread."""
    rng = np.random.default_rng(0) if rng is None else rng
    points = np.concatenate([p.points for p in populations])
    spread = float(np.sqrt(np.mean(np.var(points, axis=0))))+1e-12
    d = points.shape[1]
    out = []
    for k in range(size):
        if k % 2 == 0:
            out.append(gaussian_bump(points[rng.integers(len(points))], spread*rng.uniform(.5, 2.)))
        else:
            out.append(cosine_wave(rng.normal(size=d)/(spread*rng.uniform(.5, 2.)*np.sqrt(d)), rng.uniform(0, 2*np.pi)))
    return out


def _cutoff(r):
    """Smooth chi with chi = 1 for r <= 1, chi = 0 for r >= 2."""
    r = np.asarray(r, dtype=float)
    chi = np.ones(r.shape)
    chi[r >= 2] = 0.
    inside = (r > 1) & (r < 2)
    u = r[inside]-1
    left, right = np.exp(-1/u), np.exp(-1/(1-u))
    chi[inside] = right/(left+right)
    return chi


def truncated(observable, radius, fd_radius=1e-4):
    """The cutoff version chi(|x|/radius) Upsilon(x) of an unbounded test function
    (value only; derivatives by central differences when needed)."""
    from .operators import Observable
    return Observable(lambda x: _cutoff(np.linalg.norm(x, axis=-1)/radius)*observable.value(x), fd_radius=fd_radius)
