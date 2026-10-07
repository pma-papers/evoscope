"""EvoScope (package evoscope): see what every operator of a population-based optimizer does.

Tools that accompany the paper "Operator Calculus for Population-Based Optimization:
Modular Convergence and Finite-Population Guarantees": numerical checks of the theory's
conditions for new operators, attribution of performance to operators, algorithm-level
checks for implemented algorithms, and measured discovery guarantees.

Start with the documentation in docs/index.md.

Modules
-------
operators     operator wrappers (Transport, Reweighting, Jump, StepOperator), canonical operators, Assembly
checks        operator- and assembly-level checks (conditions A1-A4, composition, cutoff, drift coefficient)
algorithm     ask/tell interface, runs with probes, algorithm-level checks, ablation
attribution   effects of operators, drift coefficients, validation, residual form, consistency
discovery     the measured discovery bound
report        reports with verdicts and plain-language explanations
population    weighted populations
problems      the paper's test problems
testfunctions, metrics, contrasts, viz
criterion, band, modulus, runs   the evaluation criterion of the paper and the study's runner
"""
from . import (algorithm, attribution, band, checks, contrasts, criterion, discovery, metrics, modulus,
               operators, population, problems, report, runs, testfunctions)
from .operators import (Assembly, Drift, GaussianNoise, Jump, Observable, Recombination, Reweighting, Selection,
                        StepOperator, Transport, paper_assembly)
from .population import Population

__version__ = "0.2.0.dev0"
