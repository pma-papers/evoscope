"""The evolution equation of an assembly, term by term.

Through the composition theorem of the paper, an assembly of operators defines the
evolution equation

    d/dt <phi, mu_t> = G[mu_t](phi) = sum_j G_j[mu_t](phi)      for every test function phi,

one term per operator.  For the canonical operators it is the transport-reaction-jump
(TRJ) equation

    d/dt mu = -div(b mu) + (sigma^2/2) Lap mu - omega (Phi - mean Phi) mu + gamma (R[mu] - mu)
               drift       noise              selection                  recombination

This module makes the terms visible.

* A **view** is a family of test functions chosen to show one aspect of the population:
  its mean, its spread, its objective values, the mass below each objective level, or its
  density along one or two directions.  Views of moments and of objective values work in
  any dimension; density views show low-dimensional problems or projections.
* :func:`term_rates` applies every operator's generator to a view: the operator's term
  in the equation, as seen through the view.
* :func:`flow` follows the solution of the equation approximately with many particles
  (the assembly run with a large population and a small step), and :func:`ledger` adds
  up the terms along it.  By the definition of a weak solution, the time integral of the
  terms equals the change of the viewed quantity; the ledger shows how much of the change
  each operator is responsible for, and when the roles change hands.
* :func:`local_characteristics` reads off what an operator does to each individual
  (growth rate, drift, diffusion) from its effect on test functions localized at the
  individual, :func:`innovation_rate` how much mass it places away from the population,
  and :func:`fingerprint` summarizes both: which kinds of term an operator contributes,
  also for operators known only through their code.
* :class:`TRJ1D` solves the equation on a grid in one dimension, term by term, as a
  reference for the particle computations.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.special import expit

from .operators import Assembly, Jump, Observable, Recombination, Reweighting, Transport
from .population import Population

__all__ = ["View", "mean_view", "spread_view", "summary_view", "level_view", "density_view",
           "term_rates", "operator_rates", "Flow", "flow", "ledger", "merge_duplicates",
           "local_characteristics", "innovation_rate", "fingerprint", "format_fingerprint",
           "TRJ1D", "TERM_KINDS"]

_BLOCK = 4_000_000      # array entries per block in blocked evaluations
TERM_KINDS = ("reaction", "transport", "diffusion", "jump")


def _blocks(n, per_row):
    step = max(1, int(_BLOCK//max(1, per_row)))
    for start in range(0, n, step):
        yield slice(start, min(n, start+step))


def _operator_list(operators):
    if isinstance(operators, Assembly):
        return list(operators.operators)
    if isinstance(operators, (list, tuple)):
        return list(operators)
    return [operators]


def _as_function(objective, fd_radius=1e-4):
    """An object with value, gradient and Laplacian (finite differences where missing)."""
    if hasattr(objective, "value"):
        if hasattr(objective, "gradient") and hasattr(objective, "laplacian"):
            return objective
        return Observable(objective.value, fd_radius=fd_radius)
    return Observable(objective, fd_radius=fd_radius)


# ----------------------------------------------------------------------------- views
class View:
    """A family of K test functions phi_1, ..., phi_K evaluated together.

    ``value(x)`` returns an (n, K) array for points x of shape (n, d).  ``gradient(x)``
    (n, K, d) and ``laplacian(x)`` (n, K) are needed only for moves given by formula
    (:class:`~evoscope.operators.Transport` with a drift or ``sigma``); central differences
    replace them when missing.  ``labels`` names the K functions, ``axis`` holds their
    coordinates when they form a family over levels or grid points, ``shape`` the grid shape
    of a two-dimensional family, and ``title`` names the view.
    """

    def __init__(self, value, labels, gradient=None, laplacian=None, axis=None, shape=None, title="",
                 fd_radius=1e-4):
        self._value, self._gradient, self._laplacian = value, gradient, laplacian
        self.labels, self.axis, self.shape, self.title = list(labels), axis, shape, title
        self.fd_radius = fd_radius

    @property
    def size(self):
        return len(self.labels)

    def value(self, x):
        return np.asarray(self._value(np.atleast_2d(np.asarray(x, dtype=float))), dtype=float)

    def gradient(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self._gradient is not None:
            return np.asarray(self._gradient(x), dtype=float)
        r, d = self.fd_radius, x.shape[1]
        out = np.empty((len(x), self.size, d))
        for j in range(d):
            e = np.zeros(d)
            e[j] = r
            out[..., j] = (self.value(x+e)-self.value(x-e))/(2*r)
        return out

    def laplacian(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self._laplacian is not None:
            return np.asarray(self._laplacian(x), dtype=float)
        r, d = self.fd_radius, x.shape[1]
        center, total = self.value(x), np.zeros((len(x), self.size))
        for j in range(d):
            e = np.zeros(d)
            e[j] = r
            total += (self.value(x+e)-2*center+self.value(x-e))/(r*r)
        return total

    def integrate(self, points, weights):
        """sum_i weights_i phi_k(points_i) for every k (weights may be signed)."""
        points, weights = np.atleast_2d(points), np.asarray(weights, dtype=float)
        out = np.zeros(self.size)
        for block in _blocks(len(points), self.size):
            out += weights[block]@self.value(points[block])
        return out

    def tracked(self, pop):
        """<phi_k, mu> for every k: the quantities the view shows."""
        return self.integrate(pop.points, pop.weights)

    def transport_action(self, points, weights, drift, half_sigma2):
        """sum_i w_i [b(x_i) . grad phi_k(x_i) + (sigma^2/2) Lap phi_k(x_i)]: the term of a move."""
        out = np.zeros(self.size)
        d = points.shape[1]
        for block in _blocks(len(points), self.size*(d+1)):
            if drift is not None:
                out += np.einsum("i,ikd,id->k", weights[block], self.gradient(points[block]), drift[block])
            if half_sigma2:
                out += half_sigma2*(weights[block]@self.laplacian(points[block]))
        return out

    def __repr__(self):
        return f"View({self.title or 'family'}, {self.size} functions)"


def mean_view(d):
    """phi_k(x) = x_k, k = 1..d: the population mean."""
    eye = np.eye(d)
    return View(lambda x: x.copy(), [f"mean x{k+1}" for k in range(d)],
                gradient=lambda x: np.broadcast_to(eye, (len(x), d, d)),
                laplacian=lambda x: np.zeros((len(x), d)), title="mean")


def spread_view(pop, axes=None):
    """Centred second moments at ``pop``: the total spread int ||x - m||^2 dmu (the spread
    observable Var(mu) of the paper) and the variances along the principal axes of mu
    (``axes``: how many, largest first; default all when d <= 10, else 3; none when d = 1).

    The centre m and the axes are frozen at ``pop``.  The rate of the frozen function is the
    rate of the moment itself: the motion of the centre does not change a centred second
    moment to first order, and the eigenvalues of the covariance move at the rates
    u' (dSigma/dt) u along their eigenvectors u.  ``axis`` holds the principal variances."""
    m = pop.mean()
    centred = pop.points-m
    covariance = (pop.weights[:, None]*centred).T@centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    d = pop.dimension
    k = (0 if d == 1 else d if d <= 10 else 3) if axes is None else int(axes)
    U = eigenvectors[:, order[:k]]

    def value(x):
        z = x-m
        return np.column_stack([np.sum(z*z, axis=1), (z@U)**2])

    def gradient(x):
        z = x-m
        p = z@U
        return np.concatenate([2*z[:, None, :], 2*p[:, :, None]*U.T[None, :, :]], axis=1)

    def laplacian(x):
        return np.tile(np.concatenate([[2.*d], np.full(k, 2.)]), (len(x), 1))

    labels = ["total spread"]+[f"axis {j+1}" for j in range(k)]
    return View(value, labels, gradient, laplacian, axis=eigenvalues[order[:k]], title="spread")


def summary_view(objective, pop, f_star=0., target=None):
    """The error measures of the paper's convergence targets in one view: the mean gap
    int (f - f_*) dmu, the spread int ||x - m||^2 dmu (centre frozen at ``pop``) and, with a
    ``target`` minimizer, the squared distance int ||x - target||^2 dmu = W_2^2(mu, delta_target)."""
    f = _as_function(objective)
    m = pop.mean()
    t = None if target is None else np.asarray(target, dtype=float)
    d = pop.dimension
    labels = ["mean gap", "spread"]+([] if t is None else ["squared distance"])

    def value(x):
        columns = [f.value(x)-f_star, np.sum((x-m)**2, axis=1)]
        if t is not None:
            columns.append(np.sum((x-t)**2, axis=1))
        return np.column_stack(columns)

    def gradient(x):
        columns = [f.gradient(x), 2*(x-m)]
        if t is not None:
            columns.append(2*(x-t))
        return np.stack(columns, axis=1)

    def laplacian(x):
        columns = [f.laplacian(x), np.full(len(x), 2.*d)]
        if t is not None:
            columns.append(np.full(len(x), 2.*d))
        return np.column_stack(columns)

    return View(value, labels, gradient, laplacian, title="error measures")


def level_view(objective, levels, f_star=0., width=None, log=True):
    """Smoothed masses below objective levels: phi_c(x) ~ 1{f(x) - f_* <= c} for each level c.

    <phi_c, mu> is the distribution function F(c) = mu(f - f_* <= c) of the objective gaps of
    the population: the share of the population within c of the optimal value, the quantity
    p_c(t) of the paper's concentration results.  The indicator is smoothed by a logistic step
    sigma((log c - log(f - f_*))/width) (with ``log``; otherwise sigma((c - f + f_*)/width)), so
    that every operator has a well-defined term.  ``width`` defaults to the spacing of the
    levels (on the log scale with ``log``).  The term of an operator at level c is the rate at
    which it moves mass across that level: positive when it brings mass below it."""
    f = _as_function(objective)
    levels = np.asarray(levels, dtype=float)
    coordinate = np.log(levels) if log else levels
    if width is None:
        width = float(np.median(np.diff(coordinate))) if len(levels) > 1 else .25
    floor = 1e-30

    def parts(x):
        gap = f.value(x)-f_star
        if not log:
            return gap, f.gradient(x), f.laplacian(x)
        g = np.maximum(gap, floor)
        grad = f.gradient(x)
        return np.log(g), grad/g[:, None], f.laplacian(x)/g-np.sum(grad*grad, axis=1)/(g*g)

    def value(x):
        s = (np.log(np.maximum(f.value(x)-f_star, floor)) if log else f.value(x)-f_star)
        return expit((coordinate[None, :]-s[:, None])/width)

    def derivatives(x):
        s, grad_s, lap_s = parts(x)
        sig = expit((coordinate[None, :]-s[:, None])/width)
        d1 = sig*(1-sig)
        d2 = d1*(1-2*sig)
        gradient = -(d1/width)[:, :, None]*grad_s[:, None, :]
        laplacian = d2*np.sum(grad_s*grad_s, axis=1)[:, None]/width**2-d1*lap_s[:, None]/width
        return gradient, laplacian

    labels = [f"gap <= {c:.3g}" for c in levels]
    return View(value, labels, lambda x: derivatives(x)[0], lambda x: derivatives(x)[1], axis=levels,
                title="mass below level")


class _DensityView(View):
    """Gaussian-kernel density along one or two orthonormal directions (see density_view)."""

    def __init__(self, grid, bandwidth, directions):
        self.h = float(bandwidth)
        if isinstance(grid, (tuple, list)) and len(grid) == 2 and np.ndim(grid[0]) == 1:
            gx, gy = (np.asarray(g, dtype=float) for g in grid)
            X, Y = np.meshgrid(gx, gy)
            self.centres, shape, axis = np.column_stack([X.ravel(), Y.ravel()]), X.shape, (gx, gy)
            labels = [f"y=({a:.3g}, {b:.3g})" for a, b in self.centres]
        else:
            g = np.asarray(grid, dtype=float).ravel()
            self.centres, shape, axis = g[:, None], None, g
            labels = [f"y={a:.3g}" for a in g]
        self.k = self.centres.shape[1]
        if directions is not None:
            P = np.atleast_2d(np.asarray(directions, dtype=float))
            if P.shape[0] != self.k:
                raise ValueError(f"need {self.k} direction(s) for this grid")
            if self.k == 1:
                P = P/np.linalg.norm(P)
            elif not np.allclose(P@P.T, np.eye(self.k), atol=1e-8):
                raise ValueError("the two directions must be orthonormal")
            directions = P
        self.directions = directions
        self.norm = (2*np.pi*self.h*self.h)**(-self.k/2)
        super().__init__(self._kernel, labels, axis=axis, shape=shape,
                         title="density" if self.k == 1 else "density map")

    def _P(self, d):
        return np.eye(d)[:self.k] if self.directions is None else self.directions

    def _differences(self, x):
        z = x@self._P(x.shape[1]).T
        return z[:, None, :]-self.centres[None, :, :]

    def _kernel(self, x):
        diff = self._differences(x)
        return self.norm*np.exp(-np.sum(diff*diff, axis=-1)/(2*self.h*self.h))

    def gradient(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        diff = self._differences(x)
        kern = self.norm*np.exp(-np.sum(diff*diff, axis=-1)/(2*self.h*self.h))
        return (-(diff/self.h**2)*kern[..., None])@self._P(x.shape[1])

    def laplacian(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        diff = self._differences(x)
        r2 = np.sum(diff*diff, axis=-1)
        return (r2/self.h**4-self.k/self.h**2)*self.norm*np.exp(-r2/(2*self.h*self.h))

    def integrate(self, points, weights):
        points, weights = np.atleast_2d(points), np.asarray(weights, dtype=float)
        out = np.zeros(self.size)
        for block in _blocks(len(points), self.size*self.k):
            out += weights[block]@self._kernel(points[block])
        return out

    def transport_action(self, points, weights, drift, half_sigma2):
        out = np.zeros(self.size)
        P = self._P(points.shape[1])
        for block in _blocks(len(points), self.size*(self.k+1)):
            diff = self._differences(points[block])
            r2 = np.sum(diff*diff, axis=-1)
            kern = self.norm*np.exp(-r2/(2*self.h*self.h))
            if drift is not None:
                projected = drift[block]@P.T
                out += np.einsum("i,ikj,ij->k", weights[block], -(diff/self.h**2)*kern[..., None], projected)
            if half_sigma2:
                out += half_sigma2*(weights[block]@((r2/self.h**4-self.k/self.h**2)*kern))
        return out


def density_view(grid, bandwidth, directions=None):
    """Kernel-smoothed density of the population along one or two directions.

    phi_y(x) = K_h(P x - y): a Gaussian kernel of width ``bandwidth`` around each grid point
    y, with P the projection on the ``directions`` (unit vector(s); default the first one or
    two coordinate axes).  ``grid``: a 1-D array (one direction), or a pair of 1-D arrays (two
    orthonormal directions; the family runs over the product grid, ``shape`` = (len(gy),
    len(gx)) for plotting).  <phi_y, mu> is the smoothed (marginal) density at y.  The term
    of an operator is the corresponding term of the equation for this density: a signed
    function of y, positive where the operator adds mass and negative where it removes it.
    This is the weak form of the equation made visible; no density of mu is needed."""
    return _DensityView(grid, bandwidth, directions)


# ----------------------------------------------------------------------------- terms
def _recombination_offspring(op, pop, rng, exact_pairs, pairs):
    """Offspring points and weights of a Recombination: all parent pairs when there are at
    most ``exact_pairs`` (with coefficients), otherwise ``pairs`` independent draws."""
    n, k = pop.size, len(op.coefficients)
    if n*n*k <= exact_pairs:
        offspring, probs = op._offspring(pop.points)
        pair = (pop.weights[:, None]*pop.weights[None, :]).reshape(-1)
        return offspring.reshape(-1, pop.dimension), np.concatenate([p*pair for p in probs])
    first, second = pop.sample(pairs, rng), pop.sample(pairs, rng)
    xi = op.coefficients[rng.choice(k, size=pairs, p=op.probabilities)][:, None]
    return (1-xi)*pop.points[first]+xi*pop.points[second], np.full(pairs, 1/pairs)


def _jump_offspring(op, pop, rng, offspring):
    if op.law is not None:
        q = op.law(pop)
        if q.size > offspring:
            return q.points[q.sample(offspring, rng)], np.full(offspring, 1/offspring)
        return q.points, q.weights
    return np.atleast_2d(op.offspring(pop, offspring, rng)), np.full(offspring, 1/offspring)


def operator_rates(op, pop, view, rng=None, exact_pairs=500_000, pairs=200_000, offspring=50_000,
                   replicates=400, tau=None):
    """G[mu](phi_k) of one operator for every function of ``view``: its term in the equation.

    By formula for moves with a drift and/or ``sigma`` (int grad phi . b + (sigma^2/2) Lap phi),
    reweightings (-strength Cov(Phi, phi)), recombination (rate [E phi(offspring) - int phi])
    and jumps (rate [<phi, Q[mu]> - <phi, mu>]).  Recombination with more than ``exact_pairs``
    parent pairs uses ``pairs`` random pairs, and offspring samplers ``offspring`` draws.
    Any other operator is estimated from finite steps: Richardson extrapolation of the
    quotients at 2 tau and tau (default tau = 1e-3) for exact rules, and the quotient at tau
    (default 0.01) averaged over ``replicates`` steps for random rules."""
    rng = np.random.default_rng(0) if rng is None else rng
    x, w = pop.points, pop.weights
    if isinstance(op, Transport) and op.noise is None:
        drift = op._b(x, pop) if op.drift is not None else None
        return view.transport_action(x, w, drift, .5*(op.sigma or 0.)**2)
    if isinstance(op, Reweighting):
        phi = np.asarray(op._phi(x, pop), dtype=float)
        return -op.strength*view.integrate(x, w*(phi-w@phi))
    if isinstance(op, Recombination):
        points, weights = _recombination_offspring(op, pop, rng, exact_pairs, pairs)
        return op.rate*(view.integrate(points, weights)-view.integrate(x, w))
    if isinstance(op, Jump):
        points, weights = _jump_offspring(op, pop, rng, offspring)
        return op.rate*(view.integrate(points, weights)-view.integrate(x, w))
    base = view.integrate(x, w)
    if getattr(op, "exact", False):
        taus = (2e-3, 1e-3) if tau is None else (2*tau, tau)
        q = [(view.tracked(op.step(pop, t))-base)/t for t in taus]
        return 2*q[1]-q[0]
    t = .01 if tau is None else tau
    law = op.step(pop, t, rng, copies=replicates)
    return (view.tracked(law)-base)/t


def term_rates(operators, pop, view, rng=None, **kwargs):
    """Each operator's term in the evolution equation, seen through ``view``.

    ``operators``: an :class:`~evoscope.operators.Assembly`, a list of operators, or one
    operator.  Returns a dict operator name -> array (K,) of generator actions
    G_j[mu](phi_k); their sum over operators is the rate of change of <phi_k, mu> under the
    whole assembly (the composition theorem).  Keyword arguments go to :func:`operator_rates`."""
    rng = np.random.default_rng(0) if rng is None else rng
    return {op.name: operator_rates(op, pop, view, rng=rng, **kwargs) for op in _operator_list(operators)}


# ----------------------------------------------------------------------------- flows and ledgers
class Flow:
    """Populations along an approximate solution of the evolution equation: ``times`` (T,),
    ``populations`` (list of T populations), the step ``tau`` and a ``label``."""

    def __init__(self, times, populations, tau, label=""):
        self.times, self.populations, self.tau, self.label = np.asarray(times), list(populations), tau, label

    def __len__(self):
        return len(self.populations)

    def statistic(self, fn):
        """fn(population) at every recorded time, as an array."""
        return np.array([fn(pop) for pop in self.populations])

    def at(self, t):
        """The recorded population closest to time t."""
        return self.populations[int(np.argmin(np.abs(self.times-t)))]


def flow(assembly, initial, tau, t_end, rng, size=None, record=None, resample=.5):
    """Follow the evolution equation of ``assembly`` from ``initial`` with weighted particles.

    Each step of size ``tau`` applies the operators in turn to the particles: moves displace
    them, reweightings change their weights, and offspring operators replace a fraction
    rate * tau of them by offspring of parents drawn from the current weighted particles
    (each operator's ``propagate``).  The particles are resampled to equal weights when the
    effective sample size falls below ``resample`` times their number.  ``size``: number of
    particles (default: the size of ``initial``; the initial population is resampled to it).
    With many particles and a small step the populations approximate the solution mu_t at
    t = k tau: the splitting error is of order tau, the sampling error of order
    size^(-1/2), and the cost of a step is linear in the number of particles.
    ``record``: times to keep (rounded to multiples of tau; default 61 equally spaced times
    from 0 to ``t_end``)."""
    ops = _operator_list(assembly)
    steps = int(round(t_end/tau))
    if record is None:
        keep = np.unique(np.round(np.linspace(0, steps, 61)).astype(int))
    else:
        keep = np.unique(np.round(np.asarray(record, dtype=float)/tau).astype(int))
    keep = set(int(k) for k in keep if 0 <= k <= steps)
    pop = initial if isinstance(initial, Population) else Population.uniform(initial)
    size = size or pop.size
    if size != pop.size:
        pop = Population.uniform(pop.points[pop.sample(size, rng)])
    points, weights = pop.points.copy(), pop.weights.copy()
    times, populations = [], []
    if 0 in keep:
        times.append(0.)
        populations.append(pop)
    for k in range(1, steps+1):
        for op in ops:
            points, weights = op.propagate(points, weights, tau, rng)
        weights = weights/weights.sum()
        if len(points) != size or 1/np.sum(weights*weights) < resample*size:
            index = Population(points, weights).sample(size, rng)
            points, weights = points[index], np.full(size, 1/size)
        if k in keep:
            times.append(k*tau)
            populations.append(Population(points.copy(), weights.copy()))
    return Flow(times, populations, tau, getattr(assembly, "label", ""))


def _cumulative_trapezoid(values, times):
    out = np.zeros_like(values)
    dt = np.diff(times).reshape((-1,)+(1,)*(values.ndim-1))
    out[1:] = np.cumsum(.5*dt*(values[1:]+values[:-1]), axis=0)
    return out


def ledger(assembly, flow, view, rng=None, **kwargs):
    """The terms of the evolution equation along a flow, and what they add up to.

    ``view``: a :class:`View`, or a function population -> View for views centred at the
    current population (:func:`spread_view`, :func:`summary_view`).  Returns a dict with

    * ``times`` (T,), ``names`` (J operators), ``labels`` (K functions of the view);
    * ``rates`` (T, J, K): the term of each operator at each time, ``total`` (T, K) their sum;
    * ``value`` (T, K): the viewed quantity <phi_k, mu_t> along the flow, ``change`` (T, K) its
      change since the first time;
    * ``contributions`` (T, J, K): time integrals of the terms (trapezoidal rule), and
      ``predicted`` (T, K) their sum.

    For a solution of the equation, ``predicted`` equals ``change``: this is the definition
    of a weak solution.  Along a particle flow they differ by the splitting error (order
    tau) and the sampling error, which the comparison makes visible; the contributions say
    which operator is responsible for how much of the change."""
    rng = np.random.default_rng(0) if rng is None else rng
    ops = _operator_list(assembly)
    rates, values, views = [], [], []
    for pop in flow.populations:
        v = view if isinstance(view, View) else view(pop)
        rates.append([operator_rates(op, pop, v, rng=rng, **kwargs) for op in ops])
        values.append(v.tracked(pop))
        views.append(v)
    rates, values = np.array(rates), np.array(values)
    contributions = _cumulative_trapezoid(rates, flow.times)
    return {"times": flow.times, "names": [op.name for op in ops], "labels": views[-1].labels,
            "rates": rates, "total": rates.sum(axis=1), "value": values, "change": values-values[0],
            "contributions": contributions, "predicted": contributions.sum(axis=1), "views": views,
            "title": views[-1].title, "label": flow.label}


# ----------------------------------------------------------------------------- operator portraits
def merge_duplicates(pop, decimals=12):
    """The same law with repeated points merged into one atom (weights added)."""
    _, first, inverse = np.unique(np.round(pop.points, decimals), axis=0, return_index=True, return_inverse=True)
    weights = np.bincount(np.ravel(inverse), weights=pop.weights)
    return Population(pop.points[first], weights/weights.sum())


def _nearest_distances(points):
    if len(points) < 2:
        return np.ones(len(points))
    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1]


def _probe_view(centres, widths, directions):
    """psi_0, psi_1(e_k) and psi_2(u_j) localized at every centre (see local_characteristics)."""
    n, d = centres.shape
    U = np.asarray(directions, dtype=float)
    m = len(U)
    labels = ([f"psi0[{i}]" for i in range(n)]+[f"psi1[{i},{k}]" for i in range(n) for k in range(d)]
              + [f"psi2[{i},{j}]" for i in range(n) for j in range(m)])

    def parts(x):
        z = (x[:, None, :]-centres[None, :, :])/widths[None, :, None]
        r2 = np.sum(z*z, axis=-1)
        return z, r2, np.exp(-r2/2), z@U.T

    def value(x):
        z, r2, g, p = parts(x)
        return np.concatenate([(g*(1+r2/2)), (z*g[..., None]).reshape(len(x), -1),
                               (p*p*g[..., None]).reshape(len(x), -1)], axis=1)

    def gradient(x):
        z, r2, g, p = parts(x)
        h = widths[None, :, None]
        g0 = -(r2/2)[..., None]*z*g[..., None]/h
        eye = np.eye(d)
        g1 = (eye[None, None, :, :]*g[..., None, None]-z[..., :, None]*z[..., None, :]*g[..., None, None])
        g1 = g1/h[..., None]
        g2 = (2*p[..., None]*U[None, None, :, :]-(p*p)[..., None]*z[..., None, :])*g[..., None, None]
        g2 = g2/h[..., None]
        return np.concatenate([g0, g1.reshape(len(x), -1, d), g2.reshape(len(x), -1, d)], axis=1)

    def laplacian(x):
        z, r2, g, p = parts(x)
        h2 = (widths*widths)[None, :]
        l0 = -g*r2*(1+d/2-r2/2)/h2
        l1 = -z*(g*(2+d-r2))[..., None]/h2[..., None]
        l2 = g[..., None]*(2-p*p*(4+d-r2)[..., None])/h2[..., None]
        return np.concatenate([l0, l1.reshape(len(x), -1), l2.reshape(len(x), -1)], axis=1)

    return View(value, labels, gradient, laplacian, title="local probes")


def _directions(d, off_diagonal):
    eye = np.eye(d)
    rows = list(eye)
    if off_diagonal:
        rows += [(eye[k]+eye[l])/np.sqrt(2) for k in range(d) for l in range(k+1, d)]
    return np.array(rows)


def _has_formula(op):
    return (isinstance(op, Transport) and op.noise is None) or isinstance(op, (Reweighting, Jump))


def _formula_characteristics(op, pop):
    """Local characteristics of the canonical forms, from their formulas (the limit of
    vanishing probe radius)."""
    x, w = pop.points, pop.weights
    n, d = x.shape
    growth, drift, diffusion = np.zeros(n), np.zeros((n, d)), np.zeros((n, d, d))
    if isinstance(op, Transport):
        if op.drift is not None:
            drift = np.asarray(op._b(x, pop), dtype=float)
        if op.sigma:
            diffusion[:, np.arange(d), np.arange(d)] = op.sigma**2
    elif isinstance(op, Reweighting):
        phi = np.asarray(op._phi(x, pop), dtype=float)
        growth = -op.strength*(phi-w@phi)
    elif isinstance(op, Recombination):
        # offspring exactly at x: the pair (x, x) for any coefficient, and a copy of x for coefficients 0 and 1
        copy = np.isin(op.coefficients, (0., 1.))
        on_atom = np.sum(op.probabilities[copy])*w+np.sum(op.probabilities[~copy])*w*w
        growth = op.rate*(on_atom/w-1)
    else:
        q = op.law(pop) if op.law is not None else None
        if q is None:
            q = Population.uniform(op.offspring(pop, 50_000, np.random.default_rng(0)))
        distance, index = cKDTree(x).query(q.points)
        tolerance = 1e-9*max(1., float(np.max(np.abs(x))))
        on_atom = np.bincount(index[distance <= tolerance], q.weights[distance <= tolerance], minlength=n)
        growth = op.rate*(on_atom/w-1)
    return {"growth": growth, "drift": drift, "diffusion": diffusion,
            "growth_se": np.zeros(n), "drift_se": np.zeros((n, d))}


def _tracked_characteristics(op, pop, tau, replicates, radius, rng):
    """Kramers-Moyal estimates from repeated steps of a rule that moves each individual
    (the i-th point of the result is the i-th individual after the step); displacements
    longer than ``radius`` are jumps and are left out of drift and diffusion."""
    x, w = pop.points, pop.weights
    n, d = x.shape
    displacement, ratio = np.empty((replicates, n, d)), np.empty((replicates, n))
    for r in range(replicates):
        out = op.step(pop, tau, rng)
        if out.size != n:
            raise ValueError(f"{op.name}: the rule changes the number of individuals; use method='probes'")
        displacement[r], ratio[r] = out.points-x, out.weights/w
    small = (np.sum(displacement**2, axis=-1) <= radius*radius)[..., None]*displacement
    drift = small.mean(axis=0)/tau
    diffusion = np.einsum("rni,rnj->nij", small, small)/(replicates*tau)-tau*drift[:, :, None]*drift[:, None, :]
    return {"growth": (ratio.mean(axis=0)-1)/tau, "drift": drift, "diffusion": diffusion,
            "growth_se": ratio.std(axis=0, ddof=1)/(np.sqrt(replicates)*tau),
            "drift_se": small.std(axis=0, ddof=1)/(np.sqrt(replicates)*tau)}


def local_characteristics(operator, pop, method=None, radius=None, tau=.01, replicates=1000, scale=.15,
                          off_diagonal=True, rng=None, **kwargs):
    """What an operator does to each individual: growth rate, drift and diffusion.

    Near an individual x, the term of an operator acts on a test function psi through its
    value, slope and curvature at x, plus whatever mass it brings into the neighbourhood of
    x from elsewhere:

        G[mu](psi) ~ w(x) [c(x) psi(x) + b(x) . grad psi(x) + (1/2) a(x) : Hess psi(x)] + inflow.

    The per-capita growth rate c is a reaction term, the drift b a transport term and the
    diffusion matrix a a diffusion term: the local characteristics of the term at x.  The
    inflow is the jump part (see :func:`innovation_rate`).  Three ways to obtain them:

    * ``"formula"`` (default for moves with a drift and/or ``sigma``, reweightings and jumps):
      c = 0, b = drift, a = sigma^2 I for a move; c = -strength (Phi(x) - mean Phi) for a
      reweighting; c = rate (Q({x})/w(x) - 1) for offspring generation, where Q({x}) is the
      offspring mass that lands exactly on x (each individual is replaced at the rate of
      the operator; copies of it come back).
    * ``"tracking"`` (default for other operators): Kramers-Moyal estimates from
      ``replicates`` steps of size ``tau`` of a rule that moves each individual (the i-th
      output point is the i-th individual): b = E[dx]/tau and a = E[dx dx']/tau over
      displacements up to ``radius`` (default half the median distance between
      neighbours), c from the change of the weights.  Longer displacements are jumps.
      Standard errors are returned as ``growth_se`` and ``drift_se``.
    * ``"probes"``: the read-off through test functions localized at each individual (radius
      h = ``scale`` times the distance to its nearest neighbour), each with one nonzero
      Taylor coefficient at x: psi_0 = (1 + r^2/2) e^{-r^2/2} (value 1), psi_1 = (u.z) e^{-r^2/2}
      (slope u), psi_2 = (u.z)^2 e^{-r^2/2} (Hessian 2 u u'), z = (y - x)/h.  It uses only the
      law after the step and so applies to any rule, but its growth rate includes offspring
      landing within about h of x, and for rules estimated from finite steps it needs
      tau sigma^2 << h^2 (and is noisy for diffusions).  For moves and reweightings given by
      formula it reproduces the formula exactly, which is the content of the read-off.

    Repeated points are merged first.  Returns a dict with ``points``, ``weights``,
    ``growth`` (n,), ``drift`` (n, d), ``diffusion`` (n, d, d), the standard errors, and
    ``method``."""
    rng = np.random.default_rng(0) if rng is None else rng
    pop = merge_duplicates(pop)
    x, w = pop.points, pop.weights
    n, d = x.shape
    method = method or ("formula" if _has_formula(operator) else "tracking")
    if method == "formula":
        out = _formula_characteristics(operator, pop)
    elif method == "tracking":
        radius = .5*float(np.median(_nearest_distances(x))) if radius is None else radius
        out = _tracked_characteristics(operator, pop, tau, replicates, radius, rng)
    elif method == "probes":
        widths = scale*_nearest_distances(x)
        U = _directions(d, off_diagonal)
        rates = operator_rates(operator, pop, _probe_view(x, widths, U), rng=rng, **kwargs)
        r0, r1, r2 = rates[:n], rates[n:n+n*d].reshape(n, d), rates[n+n*d:].reshape(n, len(U))
        quadratic = r2*(widths*widths/w)[:, None]
        diffusion = np.zeros((n, d, d))
        diffusion[:, np.arange(d), np.arange(d)] = quadratic[:, :d]
        if off_diagonal:
            j = d
            for k in range(d):
                for l in range(k+1, d):
                    diffusion[:, k, l] = diffusion[:, l, k] = quadratic[:, j]-.5*(quadratic[:, k]+quadratic[:, l])
                    j += 1
        out = {"growth": r0/w, "drift": r1*(widths/w)[:, None], "diffusion": diffusion,
               "growth_se": np.full(n, np.nan), "drift_se": np.full((n, d), np.nan), "widths": widths}
    else:
        raise ValueError(f"unknown method {method!r}")
    return {"points": x, "weights": w, "method": method, **out}


def innovation_rate(operator, pop, radius, rng=None, exact_pairs=500_000, pairs=200_000, offspring=50_000,
                    tau=1e-3, samples=100_000):
    """The rate at which an operator places mass farther than ``radius`` from every individual.

    Zero for moves with drift and Gaussian noise (they displace mass continuously) and for
    reweighting (selection creates no new points).  For offspring generation it is the rate
    of the operator times the share of offspring landing away from the population: the part
    of the jump term that brings genuinely new points.  Other operators are measured from
    one step of size ``tau`` on about ``samples`` individuals: the mass that lands beyond
    ``radius``, per unit step.  A move with heavy-tailed noise (a Levy flight) then shows a
    positive innovation rate: it is a jump term, not a diffusion."""
    rng = np.random.default_rng(0) if rng is None else rng
    tree = cKDTree(pop.points)

    def far(points, weights):
        return float(np.sum(np.asarray(weights)[tree.query(np.atleast_2d(points))[0] > radius]))

    if (isinstance(operator, Transport) and operator.noise is None) or isinstance(operator, Reweighting):
        return 0.
    if isinstance(operator, Recombination):
        return operator.rate*far(*_recombination_offspring(operator, pop, rng, exact_pairs, pairs))
    if isinstance(operator, Jump):
        return operator.rate*far(*_jump_offspring(operator, pop, rng, offspring))
    law = operator.step(pop, tau, rng, copies=max(1, samples//pop.size))
    return far(law.points, law.weights)/tau


def fingerprint(operators, pop, objective=None, f_star=0., radius=None, rng=None, **kwargs):
    """Which kinds of term each operator contributes at ``pop``, and what they do.

    For every operator, the size of each kind of term as a rate per unit time, from
    :func:`local_characteristics` and :func:`innovation_rate`:

    * ``reaction``: the spread (weighted standard deviation) of the per-capita growth rates;
    * ``transport``: the root-mean-square drift divided by the population scale
      s = sqrt(Var(mu)/d);
    * ``diffusion``: the mean trace of the diffusion matrix divided by d s^2;
    * ``jump``: the innovation rate beyond ``radius`` (default half the median distance
      between neighbours).

    Estimates from repeated steps are corrected for their sampling noise (the squared
    standard errors are subtracted).  ``kinds`` lists the kinds above 5% of the largest and,
    for estimates, well above their noise level.
    Also the effects on the population: ``mean shift`` (rate of change of the mean, its norm
    divided by s), ``spread rate`` (of int ||x - m||^2 dmu) and, with an ``objective``,
    ``gap rate`` (of the mean gap).  Keyword arguments go to :func:`local_characteristics`.
    Returns a dict operator name -> dict; :func:`format_fingerprint` prints a table."""
    rng = np.random.default_rng(0) if rng is None else rng
    merged = merge_duplicates(pop)
    d = merged.dimension
    s = np.sqrt(max(merged.variance(), 1e-300)/d)
    radius = .5*float(np.median(_nearest_distances(merged.points))) if radius is None else radius
    views = [mean_view(d), spread_view(merged, axes=0)]
    if objective is not None:
        f = _as_function(objective)
        views.append(View(lambda x: (f.value(x)-f_star)[:, None], ["mean gap"],
                          lambda x: f.gradient(x)[:, None, :], lambda x: f.laplacian(x)[:, None]))
    out = {}
    for op in _operator_list(operators):
        chars = local_characteristics(op, merged, radius=radius, rng=rng, **kwargs)
        w, growth = chars["weights"], chars["growth"]
        trace = np.trace(chars["diffusion"], axis1=1, axis2=2)
        se_growth, se_drift = np.nan_to_num(chars["growth_se"])**2, np.nan_to_num(chars["drift_se"])**2
        reaction2 = w@((growth-w@growth)**2-se_growth)
        transport2 = w@np.sum(chars["drift"]**2-se_drift, axis=1)
        # noise level of the corrected squares (zero for formulas): a kind counts only well above it
        noise = {"reaction": 3*np.sqrt(2*np.sum(w*w*se_growth**2)),
                 "transport": 3*np.sqrt(2*np.sum((w*w)[:, None]*se_drift**2))}
        significant = {"reaction": reaction2 > noise["reaction"], "transport": transport2 > noise["transport"],
                       "diffusion": True, "jump": True}
        entry = {"reaction": float(np.sqrt(max(0., reaction2))),
                 "transport": float(np.sqrt(max(0., transport2))/s),
                 "diffusion": float(max(w@trace, 0.)/(d*s*s)),
                 "jump": innovation_rate(op, merged, radius, rng=rng),
                 "mean growth": float(w@growth), "method": chars["method"]}
        rates = [operator_rates(op, merged, v, rng=rng) for v in views]
        entry["mean shift"] = float(np.linalg.norm(rates[0])/s)
        entry["spread rate"] = float(rates[1][0])
        if objective is not None:
            entry["gap rate"] = float(rates[2][0])
        sizes = np.array([entry[k] for k in TERM_KINDS])
        entry["kinds"] = [k for k, v in zip(TERM_KINDS, sizes) if v > max(.05*sizes.max(), 1e-9) and significant[k]]
        out[op.name] = entry
    return out


def format_fingerprint(result):
    """A plain-text table of :func:`fingerprint`."""
    columns = ["reaction", "transport", "diffusion", "jump", "mean shift", "spread rate"]
    if any("gap rate" in e for e in result.values()):
        columns.append("gap rate")
    width = max(8, *(len(n) for n in result))
    lines = [f"{'operator':<{width}}"+"".join(f"{c:>12}" for c in columns)+"   kinds of term"]
    for name, e in result.items():
        lines.append(f"{name:<{width}}"+"".join(f"{e[c]:>12.3g}" for c in columns)+"   "+", ".join(e["kinds"]))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- one-dimensional solver
class TRJ1D:
    """The evolution equation of an assembly on a uniform grid in one dimension, term by term.

    The density rho on the cell centres ``grid`` is handled as a weighted population with
    atoms at the cell centres, so drifts, noise levels and selection rates that depend on the
    population are evaluated as for particles.  One term per operator (strong form):

    * move (drift b, noise sigma): -(b rho)' + (sigma^2/2) rho'' (upwind fluxes, no flux
      through the ends of the grid);
    * reweighting (rate Phi): -strength (Phi - mean Phi) rho;
    * recombination with coefficients in {0, 1/2, 1}: rate (r[rho] - rho), with r[rho] the
      density of the midpoint (X + X')/2 of independent X, X' ~ rho (coefficients 0 and 1
      copy a parent and contribute nothing);
    * jump with an offspring ``law``: rate (Q[rho] - rho), with Q binned onto the grid.

    ``terms(rho)`` returns the terms, ``solve`` integrates the equation (explicit Euler)."""

    def __init__(self, assembly, grid):
        self.operators = _operator_list(assembly)
        self.names = [op.name for op in self.operators]
        self.grid = np.asarray(grid, dtype=float)
        self.h = float(self.grid[1]-self.grid[0])
        if not np.allclose(np.diff(self.grid), self.h):
            raise ValueError("the grid must be uniform")
        for op in self.operators:
            if isinstance(op, Recombination):
                if not np.all(np.isin(op.coefficients, (0., .5, 1.))):
                    raise NotImplementedError("TRJ1D supports blending coefficients 0, 1/2 and 1 only")
            elif isinstance(op, Jump):
                if op.law is None:
                    raise NotImplementedError("TRJ1D needs offspring laws, not samplers")
            elif not (isinstance(op, Reweighting) or (isinstance(op, Transport) and op.noise is None)):
                raise NotImplementedError(f"no grid term for {op!r}; use particles (term_rates)")

    def population(self, rho):
        """The density as a weighted population on the cell centres."""
        mass = np.clip(rho, 0, None)*self.h
        return Population(self.grid[:, None], mass/mass.sum())

    def density(self, points, weights=None):
        """A population binned onto the grid (linear binning), as a density."""
        points = np.ravel(points)
        weights = np.full(len(points), 1/len(points)) if weights is None else np.asarray(weights, dtype=float)
        position = np.clip((points-self.grid[0])/self.h, 0, len(self.grid)-1)
        lower = np.minimum(np.floor(position).astype(int), len(self.grid)-2)
        upper_share = position-lower
        out = np.bincount(lower, weights*(1-upper_share), minlength=len(self.grid))
        out += np.bincount(lower+1, weights*upper_share, minlength=len(self.grid))
        return out/self.h

    def _midpoint_density(self, rho):
        p = np.clip(rho, 0, None)*self.h
        q = np.clip(fftconvolve(p, p), 0, None)        # mass at x_0 + k h/2, k = 0..2(n-1)
        out = q[0::2].copy()
        half = .5*q[1::2]
        out[:-1] += half
        out[1:] += half
        return out/self.h

    def _move(self, op, rho, pop):
        out = np.zeros_like(rho)
        if op.drift is not None:
            b = np.ravel(op._b(self.grid[:, None], pop))
            face = .5*(b[1:]+b[:-1])
            flux = np.maximum(face, 0)*rho[:-1]+np.minimum(face, 0)*rho[1:]
            flux = np.concatenate([[0.], flux, [0.]])
            out -= np.diff(flux)/self.h
        if op.sigma:
            padded = np.concatenate([[rho[0]], rho, [rho[-1]]])
            out += .5*op.sigma**2*(padded[2:]-2*rho+padded[:-2])/self.h**2
        return out

    def terms(self, rho):
        """The term of every operator at the density ``rho``: dict name -> array."""
        pop = self.population(rho)
        out = {}
        for op in self.operators:
            if isinstance(op, Transport):
                term = self._move(op, rho, pop)
            elif isinstance(op, Reweighting):
                phi = np.ravel(op._phi(self.grid[:, None], pop))
                term = -op.strength*(phi-pop.weights@phi)*rho
            elif isinstance(op, Recombination):
                share = float(np.sum(op.probabilities[op.coefficients == .5]))
                term = op.rate*share*(self._midpoint_density(rho)-rho)
            else:
                q = op.law(pop)
                term = op.rate*(self.density(q.points, q.weights)-rho)
            out[op.name] = term
        return out

    def rate(self, rho):
        return sum(self.terms(rho).values())

    def stable_step(self, rho, cfl=.4):
        """A step size for explicit Euler from the coefficients at ``rho``."""
        pop = self.population(rho)
        limits = [np.inf]
        for op in self.operators:
            if isinstance(op, Transport):
                if op.drift is not None:
                    limits.append(self.h/max(1e-12, np.max(np.abs(op._b(self.grid[:, None], pop)))))
                if op.sigma:
                    limits.append(self.h**2/op.sigma**2)
            elif isinstance(op, Reweighting):
                phi = np.ravel(op._phi(self.grid[:, None], pop))
                limits.append(1/max(1e-12, op.strength*(phi.max()-phi.min())))
            else:
                limits.append(1/max(1e-12, op.rate))
        return cfl*min(limits)

    def solve(self, rho0, t_end, record=None, dt=None):
        """Integrate d rho/dt = sum of the terms from ``rho0`` to ``t_end`` (explicit Euler).

        ``record``: times at which to keep the density (default 61 equally spaced times).
        Returns a dict with ``times`` (T,), ``densities`` (T, G) and the step ``dt``."""
        rho = np.asarray(rho0, dtype=float)
        rho = rho/(rho.sum()*self.h)
        dt = self.stable_step(rho) if dt is None else dt
        steps = int(np.ceil(t_end/dt))
        dt = t_end/steps
        record = np.linspace(0, t_end, 61) if record is None else np.asarray(record, dtype=float)
        keep = {int(round(t/dt)): t for t in record}
        times, densities = [], []
        for k in range(steps+1):
            if k in keep:
                times.append(k*dt)
                densities.append(rho.copy())
            if k < steps:
                rho = np.clip(rho+dt*self.rate(rho), 0, None)
                rho = rho/(rho.sum()*self.h)
        return {"times": np.array(times), "densities": np.array(densities), "dt": dt}
