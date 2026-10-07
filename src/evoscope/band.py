"""Level-band constants of the evaluation-complexity criterion of the paper.

The criterion needs, on the band {|f - c| <= eta} around the target level c:
L_{T,eps} (largest gradient norm, so that f is Lipschitz there), r_0 = eta / L, and a
bound M on the density of the objective values of the search law.  One route to M is
the coarea bound M = rho-bar H / kappa with kappa the smallest gradient norm on the band
and H the largest (d-1)-dimensional area of a level set in the band.  These functions
estimate the constants by sampling the band from a box, as in the worked example of
the paper; a sample minimum overestimates kappa, so reduce it by a margin before use.
"""
import math

import numpy as np

__all__ = ["band_constants", "gaussian_density_bound", "coarea_density_bound", "empirical_density_bound"]


def band_constants(objective, gradient, level, eta, low, high, dimension, samples, rng,
                   levels=11, half_width=.005):
    """kappa, L, r_0 and H on the band {|f - level| <= eta}, from `samples` uniform draws
    in the box [low, high]^dimension (low and high scalars or arrays of length `dimension`).

    H is the largest coarea estimate (volume of {|f - u| <= half_width} / (2 half_width)
    times the mean gradient norm there) over `levels` equally spaced u in the band."""
    points = rng.uniform(low, high, size=(samples, dimension))
    values = objective(points)
    band = points[(values >= level-eta) & (values <= level+eta)]
    if len(band) == 0:
        raise ValueError("no sample fell in the band: enlarge the box, the band, or the sample")
    norms = np.linalg.norm(gradient(band), axis=1)
    kappa, L = float(norms.min()), float(norms.max())
    volume = float(np.prod(np.broadcast_to(np.asarray(high, float)-np.asarray(low, float), (dimension,))))
    areas = []
    for u in np.linspace(level-eta, level+eta, levels):
        selected = (values >= u-half_width) & (values <= u+half_width)
        areas.append(selected.mean()*volume/(2*half_width)*np.linalg.norm(gradient(points[selected]), axis=1).mean())
    return {"kappa": kappa, "L": L, "r0": eta/L, "H": float(max(areas)), "band_samples": int(len(band))}


def gaussian_density_bound(sigma0, d):
    """Density bound (2 pi sigma0^2)^(-d/2) of any Gaussian mixture with covariances >= sigma0^2 I."""
    return (2*math.pi*sigma0**2)**(-d/2)


def coarea_density_bound(rho_bar, H, kappa):
    """M = rho-bar H / kappa: worst-case bound on the objective-value density (the coarea bound of the paper)."""
    return rho_bar*H/kappa


def empirical_density_bound(values, level, eta, bins=20):
    """Largest objective-value density over `bins` equal sub-bands of the band, from the
    objective values of populations (a histogram supremum, not the band average)."""
    values = np.ravel(values)
    counts, _ = np.histogram(values, bins=bins, range=(level-eta, level+eta))
    return float(counts.max()/len(values)/(2*eta/bins))
