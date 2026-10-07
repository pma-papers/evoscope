"""Operators and assemblies: canonical ones, and wrappers for your own.

An *operator* is one population-update rule with a step size tau: it maps a population
law mu to a new law T_tau mu, and is close to doing nothing when tau is small.  The theory
needs two things from it:

* ``step(pop, tau, rng)``: one application of the rule.  For selection, recombination
  with a finite set of blending coefficients, and deterministic drift, the result is the
  exact new law (a weighted population); for rules with random perturbations it is one
  random realization.
* ``generator(pop, obs)``: the first-order change of the population average of a test
  function per unit step, G[mu](Upsilon) = lim (<Upsilon, T_tau mu> - <Upsilon, mu>)/tau.
  It is computed by formula where one is known and estimated from finite steps otherwise,
  so your own operators need no derivatives and no formulas.

Your own operator
-----------------
Wrap the rule in the form closest to how you wrote it:

* :class:`Transport`: moves each individual, x -> x + tau b(x) + sqrt(tau) xi (mutations,
  gradient-like steps, attraction to a consensus point, ...);
* :class:`Reweighting`: changes the weights of individuals by exp(-tau Phi(x)) (selection
  of any kind, including rank-based);
* :class:`Jump`: replaces a fraction tau of the population by new individuals generated
  from the current population (crossover, differential variation, restarts, any
  offspring-generating mechanism);
* :class:`StepOperator`: any rule given directly as a function of (population, tau, rng).

The rules may depend on the whole population (pass functions of ``(x, population)``).

An :class:`Assembly` applies operators in a given order; its generator is the sum of the
component generators (the composition theorem of the paper), which is what
attributes the first-order change of an error measure to the individual operators.
"""
import inspect

import numpy as np

from .population import Population

__all__ = ["Observable", "Operator", "Transport", "Reweighting", "Jump", "StepOperator",
           "Drift", "GaussianNoise", "Selection", "Recombination", "Assembly", "estimate_generator",
           "normalized_fd_drift", "objective_rate", "paper_assembly", "PAPER_SWITCHES"]


def _takes_population(fn, plain_arguments):
    """Whether ``fn`` expects the population as an extra positional argument."""
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    except (TypeError, ValueError):
        return False
    return len(params) > plain_arguments


# ----------------------------------------------------------------------------- test functions
class Observable:
    """A test function Upsilon with value, gradient and Laplacian.

    Missing derivatives are replaced by central differences with radius ``fd_radius``.
    ``gaussian_mean(mean, variance)`` (optional) returns E Upsilon(mean + sqrt(variance) Z)
    and makes finite steps with Gaussian noise exact.
    """

    def __init__(self, value, gradient=None, laplacian=None, gaussian_mean=None, fd_radius=1e-4):
        self._value, self._gradient, self._laplacian = value, gradient, laplacian
        self.fd_radius = fd_radius
        if gaussian_mean is not None:
            self.gaussian_mean = gaussian_mean

    def value(self, x):
        return self._value(np.asarray(x, dtype=float))

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        if self._gradient is not None:
            return self._gradient(x)
        r, out = self.fd_radius, np.empty_like(x)
        for j in range(x.shape[-1]):
            e = np.zeros(x.shape[-1]); e[j] = r
            out[..., j] = (self._value(x+e)-self._value(x-e))/(2*r)
        return out

    def laplacian(self, x):
        x = np.asarray(x, dtype=float)
        if self._laplacian is not None:
            return self._laplacian(x)
        r, center, total = self.fd_radius, self._value(x), 0.
        for j in range(x.shape[-1]):
            e = np.zeros(x.shape[-1]); e[j] = r
            total = total+(self._value(x+e)-2*center+self._value(x-e))/(r*r)
        return total

    @classmethod
    def squared_distance(cls, point):
        """Upsilon(x) = ||x - point||^2, the quadratic gauge of the paper."""
        point = np.asarray(point, dtype=float)
        return cls(lambda x: np.sum((x-point)**2, axis=-1), lambda x: 2*(x-point),
                   lambda x: np.full(x.shape[:-1], 2.*x.shape[-1]),
                   lambda m, v: np.sum((m-point)**2, axis=-1)+v*m.shape[-1])


# ----------------------------------------------------------------------------- base class
class Operator:
    """Base class.  Subclasses implement ``step``; ``generator`` falls back to estimation."""
    name = "?"
    exact = False        # step() returns the exact law of T_tau mu (finite support)
    pointwise = False    # acts on each individual separately (can be applied to sampled points exactly)

    @property
    def finite_support(self):
        return self.exact

    def step(self, pop, tau, rng=None, copies=1):
        raise NotImplementedError

    def finite_step(self, pop, tau):
        if not self.exact:
            raise NotImplementedError(f"{self.name}: no exact finite step; use step() with an rng")
        return self.step(pop, tau)

    def generator(self, pop, obs, rng=None, **kwargs):
        """G[mu](obs), estimated from finite steps (see :func:`estimate_generator`)."""
        return estimate_generator(self, pop, obs, rng=rng, **kwargs)["estimate"]

    def propagate(self, points, weights, tau, rng):
        """Apply the operator to a weighted sample of individuals; returns (points, weights)."""
        law = self.step(Population(points, weights/weights.sum()), tau, rng)
        return law.points, law.weights

    def sample_after(self, points, tau, rng):
        p, w = self.propagate(points, np.full(len(points), 1/len(points)), tau, rng)
        return p if len(p) == len(points) and np.allclose(w, w[0]) else p[Population(p, w).sample(len(points), rng)]

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"


# ----------------------------------------------------------------------------- transport (mutation)
class Transport(Operator):
    """Moves each individual: x -> x + tau b(x) + sqrt(tau) xi.

    ``drift``: function of x (or of x and the population) returning b, shape like x.
    ``noise``: function of (x, rng) (or (x, population, rng)) returning increments xi with
    mean zero, shape like x; or ``sigma`` for isotropic Gaussian increments sigma Z.
    With a drift and/or ``sigma`` the generator is computed by formula
    (int grad(Upsilon) . b + (sigma^2/2) Lap(Upsilon) dmu); with a general noise it is estimated.
    """
    pointwise = True

    def __init__(self, drift=None, noise=None, sigma=None, name="M"):
        if noise is not None and sigma is not None:
            raise ValueError("give either noise or sigma")
        self.drift, self.noise, self.sigma, self.name = drift, noise, sigma, name
        self._drift_pop = drift is not None and _takes_population(drift, 1)
        self._noise_pop = noise is not None and _takes_population(noise, 2)
        self.exact = noise is None and not sigma

    def _b(self, x, pop):
        if self.drift is None:
            return np.zeros_like(x)
        return self.drift(x, pop) if self._drift_pop else self.drift(x)

    def _xi(self, x, pop, rng):
        if self.noise is not None:
            return self.noise(x, pop, rng) if self._noise_pop else self.noise(x, rng)
        return (self.sigma or 0.)*rng.normal(size=x.shape)

    def step(self, pop, tau, rng=None, copies=1):
        if self.exact:
            return Population(pop.points+tau*self._b(pop.points, pop), pop.weights)
        x = np.repeat(pop.points, copies, axis=0)
        moved = x+tau*self._b(x, pop)+np.sqrt(tau)*self._xi(x, pop, rng)
        return Population(moved, np.repeat(pop.weights, copies)/copies)

    def propagate(self, points, weights, tau, rng):
        pop = Population(points, weights/weights.sum())
        moved = points+tau*self._b(points, pop)
        if not self.exact:
            moved = moved+np.sqrt(tau)*self._xi(points, pop, rng)
        return moved, weights

    def generator(self, pop, obs, rng=None, **kwargs):
        if self.noise is not None:
            return super().generator(pop, obs, rng=rng, **kwargs)
        value = float(np.sum(pop.weights*np.sum(self._b(pop.points, pop)*obs.gradient(pop.points), axis=-1)))
        if self.sigma:
            value += self.sigma*self.sigma*(.5*np.sum(pop.weights*obs.laplacian(pop.points)))
        return value


class Drift(Transport):
    """Deterministic motion x -> x + tau b(x).  Generator: int grad(Upsilon) . b dmu."""

    def __init__(self, field, name="D"):
        super().__init__(drift=field, name=name)
        self.field = field

    def generator(self, pop, obs, rng=None, **kwargs):
        b = self._b(pop.points, pop)
        return float(np.sum(pop.weights*np.sum(b*obs.gradient(pop.points), axis=-1)))

    def finite_step(self, pop, tau):
        return Population(pop.points+tau*self._b(pop.points, pop), pop.weights)


class GaussianNoise(Transport):
    """Isotropic Gaussian mutation x -> x + sigma sqrt(tau) Z.  Generator: (sigma^2/2) int Lap(Upsilon) dmu."""

    def __init__(self, sigma, name="H"):
        super().__init__(sigma=float(sigma), name=name)
        self.exact = False

    def generator(self, pop, obs, rng=None, **kwargs):
        return float(self.sigma*self.sigma*(.5*np.sum(pop.weights*obs.laplacian(pop.points))))

    def step(self, pop, tau, rng=None, copies=1):
        if self.sigma == 0:
            return pop
        return super().step(pop, tau, rng, copies)

    def sample_after(self, points, tau, rng):
        return points+self.sigma*np.sqrt(tau)*rng.normal(size=points.shape)

    def propagate(self, points, weights, tau, rng):
        return self.sample_after(points, tau, rng), weights


# ----------------------------------------------------------------------------- reweighting (selection)
class Reweighting(Operator):
    """Balanced reweighting: weights proportional to w exp(-tau strength Phi(x)).

    ``rate``: function of x (or of x and the population, e.g. for rank-based selection)
    returning Phi; lower Phi is favoured.  Generator: -strength Cov_mu(Phi, Upsilon)."""
    exact = True

    def __init__(self, rate, strength=1., name="S"):
        self.rate, self.strength, self.name = rate, float(strength), name
        self._rate_pop = _takes_population(rate, 1)

    def _phi(self, x, pop):
        return self.rate(x, pop) if self._rate_pop else self.rate(x)

    def generator(self, pop, obs, rng=None, **kwargs):
        w, phi, u = pop.weights, self._phi(pop.points, pop), obs.value(pop.points)
        return float(self.strength*(-(np.sum(w*phi*u)-np.sum(w*phi)*np.sum(w*u))))

    def step(self, pop, tau, rng=None, copies=1):
        weights = pop.weights*np.exp(-tau*self.strength*self._phi(pop.points, pop))
        return Population(pop.points, weights/weights.sum())

    def propagate(self, points, weights, tau, rng):
        pop = Population(points, weights/weights.sum())
        return points, weights*np.exp(-tau*self.strength*self._phi(points, pop))


class Selection(Reweighting):
    """Selection by a rate Phi (alias of :class:`Reweighting`; the paper's selection uses Phi = f/(1+f))."""


# ----------------------------------------------------------------------------- jumps (offspring)
class Jump(Operator):
    """Replaces a fraction rate*tau of the population by offspring generated from it:
    T_tau mu = (1 - rate tau) mu + rate tau Q[mu].

    ``offspring``: function of (population, size, rng) returning `size` new individuals
    drawn from Q[mu] (any crossover or variation mechanism);
    ``law``: alternatively, a function of the population returning Q[mu] exactly as a
    weighted :class:`Population` (finite offspring sets).  With ``law`` the step and the
    generator rate (<Upsilon, Q[mu]> - <Upsilon, mu>) are exact; with ``offspring`` they are
    estimated from ``offspring_samples`` draws per step."""

    def __init__(self, offspring=None, law=None, rate=1., name="J", offspring_samples=None):
        if (offspring is None) == (law is None):
            raise ValueError("give exactly one of offspring or law")
        self.offspring, self.law, self.rate, self.name = offspring, law, float(rate), name
        self.offspring_samples = offspring_samples
        self.exact = law is not None

    def _offspring_law(self, pop, rng, copies=1):
        if self.law is not None:
            return self.law(pop)
        size = (self.offspring_samples or pop.size)*copies
        return Population.uniform(self.offspring(pop, size, rng))

    def step(self, pop, tau, rng=None, copies=1):
        if self.rate*tau > 1:
            raise ValueError("rate * tau must not exceed one")
        q = self._offspring_law(pop, rng, copies)
        return Population(np.concatenate([pop.points, q.points]),
                          np.concatenate([(1-self.rate*tau)*pop.weights, self.rate*tau*q.weights]))

    def generator(self, pop, obs, rng=None, **kwargs):
        if self.law is not None:
            q = self.law(pop)
            return float(self.rate*(q.expect(obs.value(q.points))-pop.expect(obs.value(pop.points))))
        return super().generator(pop, obs, rng=rng, **kwargs)

    def propagate(self, points, weights, tau, rng):
        pop = Population(points, weights/weights.sum())
        replace = rng.random(len(points)) < self.rate*tau
        if replace.any():
            points = points.copy()
            if self.law is not None:
                q = self.law(pop)
                points[replace] = q.points[q.sample(int(replace.sum()), rng)]
            else:
                points[replace] = self.offspring(pop, int(replace.sum()), rng)
        return points, weights


class Recombination(Jump):
    """Offspring (1 - xi) x + xi y of independent parents x, y ~ mu, with blending
    coefficients xi from a finite distribution (default: the midpoint, xi = 1/2), replacing
    a fraction rate*tau of the population.
    Generator: rate [E Upsilon((1 - xi) X + xi X') - int Upsilon dmu]."""

    def __init__(self, rate=1., coefficients=(.5,), probabilities=None, name="R"):
        self.coefficients = np.atleast_1d(np.asarray(coefficients, dtype=float))
        k = len(self.coefficients)
        self.probabilities = np.full(k, 1/k) if probabilities is None else np.asarray(probabilities, dtype=float)
        super().__init__(law=self._pair_law, rate=rate, name=name)

    def _offspring(self, points):
        x, y = points[:, None, :], points[None, :, :]
        out = [.5*(x+y) if xi == .5 else (1-xi)*x+xi*y for xi in self.coefficients]
        return np.stack(out), self.probabilities

    def _pair_law(self, pop):
        offspring, probs = self._offspring(pop.points)
        n, d = pop.points.shape
        pair = (pop.weights[:, None]*pop.weights[None, :]).reshape(n*n)
        return Population(offspring.reshape(-1, d), np.concatenate([p*pair for p in probs]))

    def generator(self, pop, obs, rng=None, **kwargs):
        offspring, probs = self._offspring(pop.points)
        pair = pop.weights[:, None]*pop.weights[None, :]
        expected = sum(p*np.sum(pair*obs.value(o)) for p, o in zip(probs, offspring))
        return float(self.rate*(expected-np.sum(pop.weights*obs.value(pop.points))))

    def step(self, pop, tau, rng=None, copies=1):
        if self.rate*tau > 1:
            raise ValueError("rate * tau must not exceed one")
        offspring, probs = self._offspring(pop.points)
        n, d = pop.points.shape
        pair = (pop.weights[:, None]*pop.weights[None, :]).reshape(n*n)
        points = np.concatenate([pop.points, offspring.reshape(-1, d)])
        weights = np.concatenate([(1-self.rate*tau)*pop.weights,
                                  self.rate*tau*np.concatenate([p*pair for p in probs])])
        return Population(points, weights)

    def propagate(self, points, weights, tau, rng):
        pop = Population(points, weights/weights.sum())
        replace = rng.random(len(points)) < self.rate*tau
        if replace.any():
            k = int(replace.sum())
            first, second = pop.sample(k, rng), pop.sample(k, rng)
            xi = self.coefficients[rng.choice(len(self.coefficients), size=k, p=self.probabilities)][:, None]
            points = points.copy()
            points[replace] = (1-xi)*points[first]+xi*points[second]
        return points, weights


# ----------------------------------------------------------------------------- arbitrary rules
class StepOperator(Operator):
    """Any update rule given as ``step(population, tau, rng)`` returning a
    :class:`Population` or an array of points (equal weights).  Set ``exact=True`` if the
    rule is deterministic and returns the exact new law."""

    def __init__(self, step, name="T", exact=False):
        self._step, self.name, self.exact = step, name, exact

    def step(self, pop, tau, rng=None, copies=1):
        if copies == 1 or self.exact:
            out = self._step(pop, tau, rng)
            return out if isinstance(out, Population) else Population.uniform(out)
        laws = [self.step(pop, tau, rng) for _ in range(copies)]
        return Population(np.concatenate([q.points for q in laws]), np.concatenate([q.weights for q in laws])/copies)


# ----------------------------------------------------------------------------- generator estimation
def _mean_after(rule, pop, tau, obs, rng):
    if isinstance(rule, Assembly):
        return rule.finite_step_mean(pop, tau, obs, rng=rng)
    law = rule.step(pop, tau, rng)
    return law.expect(obs.value(law.points))


def estimate_generator(rule, pop, obs, rng=None, taus=None, replicates=200, seed=0):
    """Estimate G[mu](obs) for an operator or assembly from finite steps.

    Exact rules (finite-support laws): quotients at taus (default 2e-3, 1e-3) and the
    Richardson extrapolation 2 Q(tau/2) - Q(tau); standard error zero.
    Random rules: `replicates` independent steps per tau, with common random numbers across
    the taus (default 0.02, 0.01); the estimate is the mean quotient at the smallest tau and
    ``se`` its standard error.  Returns a dict with ``estimate``, ``se``, ``taus``, ``quotients``.
    """
    exact = getattr(rule, "exact", False) or (isinstance(rule, Assembly) and rule.exact_for(obs))
    base = pop.expect(obs.value(pop.points))
    if exact:
        taus = tuple(taus or (2e-3, 1e-3))
        q = np.array([(_mean_after(rule, pop, t, obs, None)-base)/t for t in taus])
        estimate = 2*q[-1]-q[-2] if len(q) > 1 else q[-1]
        return {"estimate": float(estimate), "se": 0., "taus": np.array(taus), "quotients": q}
    taus = tuple(taus or (.02, .01))
    seed = int(np.random.default_rng(seed).integers(2**31)) if rng is None else int(rng.integers(2**31))
    q = np.array([[(_mean_after(rule, pop, t, obs, np.random.default_rng([seed, k]))-base)/t
                   for k in range(replicates)] for t in taus])
    return {"estimate": float(q[-1].mean()), "se": float(q[-1].std(ddof=1)/np.sqrt(replicates)),
            "taus": np.array(taus), "quotients": q.mean(axis=1), "quotient_se": q.std(axis=1, ddof=1)/np.sqrt(replicates)}


# ----------------------------------------------------------------------------- mixtures (for sampling)
class _Mixture:
    """A finite-support law kept as a mixture of components, so that draws for different step
    sizes use the same random numbers in the same way (first the component, then the atom)."""

    def __init__(self, parts):
        self.parts = [(p, float(m)) for p, m in parts if m > 0]

    def flatten(self):
        return Population(np.concatenate([p.points for p, _ in self.parts]),
                          np.concatenate([m*p.weights for p, m in self.parts]))

    def apply(self, op, tau):
        if isinstance(op, Jump):
            if op.rate*tau > 1:
                raise ValueError("rate * tau must not exceed one")
            q = op._offspring_law(self.flatten(), None) if op.law is not None else None
            if q is None:
                return _Mixture([(op.step(self.flatten(), tau), 1.)])
            return _Mixture([(p, m*(1-op.rate*tau)) for p, m in self.parts]+[(q, op.rate*tau)])
        if isinstance(op, Reweighting):
            whole = self.flatten()
            parts = []
            for p, m in self.parts:
                w = p.weights*np.exp(-tau*op.strength*op._phi(p.points, whole))
                parts.append((Population(p.points, w/w.sum()), m*w.sum()))
            total = sum(m for _, m in parts)
            return _Mixture([(p, m/total) for p, m in parts])
        if isinstance(op, Transport) and op.exact:
            whole = self.flatten()
            return _Mixture([(Population(p.points+tau*op._b(p.points, whole), p.weights), m) for p, m in self.parts])
        return _Mixture([(op.finite_step(self.flatten(), tau), 1.)])

    def sample(self, size, rng):
        u, v = rng.random(size), rng.random(size)
        cum = np.cumsum([m for _, m in self.parts])
        cum[-1] = 1.
        which = np.searchsorted(cum, u, side="right")
        out = np.empty((size, self.parts[0][0].dimension))
        for c, (p, _) in enumerate(self.parts):
            sel = which == c
            if sel.any():
                cdf = np.cumsum(p.weights)
                cdf[-1] = 1.
                out[sel] = p.points[np.searchsorted(cdf, v[sel], side="right")]
        return out


# ----------------------------------------------------------------------------- assemblies
class Assembly:
    """Operators applied in list order (the first entry acts first).

    ``Assembly([S, R, D, H])`` is the composition H_tau D_tau R_tau S_tau, which the paper
    writes M_tau R_tau S_tau with the mutation M_tau = H_tau D_tau.
    """

    def __init__(self, operators, label=None):
        self.operators = list(operators)
        self.label = label or " ".join(op.name for op in reversed(self.operators))

    @property
    def names(self):
        return [op.name for op in self.operators]

    @property
    def exact(self):
        return all(op.exact for op in self.operators)

    def exact_for(self, obs):
        """Whether finite_step_mean is exact for this test function."""
        _, rest = self._finite_part_ops()
        return all(isinstance(op, GaussianNoise) for op in rest) and (
            hasattr(obs, "gaussian_mean") or all(op.sigma == 0 for op in rest))

    def generator_actions(self, pop, obs, rng=None, **kwargs):
        """Component actions G_j[mu](Upsilon), keyed by operator name."""
        return {op.name: op.generator(pop, obs, rng=rng, **kwargs) for op in self.operators}

    def generator(self, pop, obs, rng=None, **kwargs):
        """The composite pre-generator: the sum of the component actions (composition theorem)."""
        return sum(self.generator_actions(pop, obs, rng=rng, **kwargs).values())

    def _finite_part_ops(self):
        ops = list(self.operators)
        prefix = []
        while ops and ops[0].exact:
            prefix.append(ops.pop(0))
        return prefix, ops

    def _finite_part(self, pop, tau):
        """Apply the leading exact operators; return the law and the remaining operators."""
        law, rest = pop, list(self.operators)
        while rest and rest[0].exact:
            law = rest.pop(0).finite_step(law, tau)
        return law, rest

    def step(self, pop, tau, rng=None, copies=1):
        """One application of every operator in turn (exact law if every operator is exact)."""
        law = pop
        for op in self.operators:
            law = op.step(law, tau, rng, copies)
        return law

    def finite_step_mean(self, pop, tau, obs, rng=None, samples=200_000):
        """int Upsilon d(T_tau mu): exact when every random operator after the exact prefix
        is Gaussian noise and the test function has ``gaussian_mean``; otherwise a Monte
        Carlo estimate from `samples` draws (needs ``rng``)."""
        law, rest = self._finite_part(pop, tau)
        if all(isinstance(op, GaussianNoise) for op in rest):
            variance = sum(tau*op.sigma*op.sigma for op in rest)
            if hasattr(obs, "gaussian_mean"):
                return float(np.sum(law.weights*obs.gaussian_mean(law.points, variance)))
            if variance == 0:
                return float(np.sum(law.weights*obs.value(law.points)))
        if rng is None:
            raise ValueError("an exact finite step is not available here; pass rng for a Monte Carlo estimate")
        return float(np.mean(obs.value(self.sample(pop, tau, samples, rng))))

    def finite_step_quotient(self, pop, tau, obs, **kwargs):
        """D_tau = [int Upsilon d(T_tau mu) - int Upsilon dmu] / tau, the finite-step change per unit step."""
        return (self.finite_step_mean(pop, tau, obs, **kwargs)-pop.expect(obs.value(pop.points)))/tau

    def proposer(self, tau):
        """The proposal function ``(population, size, rng) -> points`` of one step of size tau,
        as expected by :func:`evoscope.runs.run` and :class:`evoscope.algorithm.AssemblyAlgorithm`."""
        return lambda pop, size, rng: self.sample(pop, tau, size, rng)

    def sample(self, pop, tau, size, rng, oversample=2048):
        """`size` draws from T_tau mu (the candidates of one generation).

        Exact (independent draws from T_tau mu) when every operator after the leading exact
        ones acts on individuals separately (transport, noise).  Otherwise the remaining
        operators act on an intermediate sample of max(size, oversample) individuals, which
        approximates the population-level effect of selection and offspring generation."""
        prefix, rest = self._finite_part_ops()
        mixture = _Mixture([(pop, 1.)])
        for op in prefix:
            mixture = mixture.apply(op, tau)
        if all(op.pointwise for op in rest):
            points = mixture.sample(size, rng)
            for op in rest:
                points = op.sample_after(points, tau, rng)
            return points
        n0 = max(size, oversample)
        points, weights = mixture.sample(n0, rng), np.full(n0, 1/n0)
        for op in rest:
            points, weights = op.propagate(points, weights, tau, rng)
        final = Population(points, weights/weights.sum())
        return points[final.sample(size, rng)]


# ----------------------------------------------------------------------------- the study's operators
def _smooth_cutoff(x, radius):
    relative = np.linalg.norm(x, axis=-1)/radius
    chi = np.ones(relative.shape)
    chi[relative >= 2] = 0.
    inside = (relative > 1) & (relative < 2)
    u = relative[inside]-1
    left, right = np.exp(-1/u), np.exp(-1/(1-u))
    chi[inside] = right/(left+right)
    return chi


def normalized_fd_drift(objective, radius=1e-3, cutoff_radius=None):
    """The derivative-free drift of the study: b = -chi g / sqrt(1 + ||g||^2), with g the
    central-difference field of radius `radius` and chi a smooth cutoff equal to one on the
    ball of radius `cutoff_radius` (default 10 sqrt(d)) and zero outside twice that radius."""
    def field(x):
        x = np.asarray(x, dtype=float)
        d = x.shape[-1]
        g = np.zeros_like(x)
        for j in range(d):
            e = np.zeros(d); e[j] = radius
            g[..., j] = (objective(x+e)-objective(x-e))/(2*radius)
        chi = _smooth_cutoff(x, 10*np.sqrt(d) if cutoff_radius is None else cutoff_radius)
        return -chi[..., None]*g/np.sqrt(1+np.sum(g*g, axis=-1))[..., None]
    return field


def objective_rate(objective, transform=lambda u: u/(1+u)):
    """Selection rate Phi(x) = transform(f(x)); the study uses f/(1+f)."""
    return lambda x: transform(objective(x))


PAPER_SWITCHES = [(omega, gamma, sigma) for omega in (0, 1) for gamma in (0, 1) for sigma in (0., .2)]


def paper_assembly(problem, omega=1, gamma=1, sigma=.2, order="MRS"):
    """One of the eight assemblies of the paper's numerical study, with switches
    omega (selection), gamma (recombination), sigma (noise) and order "MRS" (S first)
    or "MSR" (R first).  Switched-off components stay in the assembly with zero action."""
    S = Selection(objective_rate(problem.value), strength=omega)
    R = Recombination(rate=gamma)
    D = Drift(normalized_fd_drift(problem.value))
    H = GaussianNoise(sigma)
    first = [S, R] if order == "MRS" else [R, S]
    label = "".join(n for n, on in (("S", omega), ("R", gamma), ("H", sigma)) if on) or "D only"
    return Assembly(first+[D, H], label=f"{label} ({order})")
