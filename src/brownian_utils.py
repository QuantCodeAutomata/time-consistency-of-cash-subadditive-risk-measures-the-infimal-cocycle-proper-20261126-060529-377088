"""Shared Monte Carlo conditional-expectation utilities.

The paper's duality formulas condition on F_s, F_u (Section 2). On a
Brownian filtration with s possibly strictly positive and non-trivial,
``E[Y | F_s]`` is estimated path-by-path by a least-squares (Longstaff-
Schwartz style) regression of ``Y`` on a polynomial basis of ``W_s``
(custom implementation: no closed-form conditional-expectation estimator
is available for a generic position Y, and this is the standard
regression-Monte-Carlo approach used throughout the BSDE numerics
literature referenced in Section 4 of the paper).
"""

from __future__ import annotations

import numpy as np


def polynomial_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Design matrix of monomials ``1, x, x**2, ..., x**degree`` per column of x.

    Parameters
    ----------
    x:
        Array of shape ``(n_paths, dim)`` with the conditioning variables
        (e.g. ``W_s``).
    degree:
        Maximum total polynomial degree per coordinate (monomials are not
        cross-multiplied across coordinates, to keep the basis size linear
        in ``dim``).

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(n_paths, 1 + dim * degree)``.
    """
    x = np.atleast_2d(x)
    if x.shape[0] == 1 and x.ndim == 2 and x.shape[1] != 1:
        x = x.T if x.shape[1] > 1 else x
    n = x.shape[0]
    cols = [np.ones(n)]
    for d in range(x.shape[1]):
        for p in range(1, degree + 1):
            cols.append(x[:, d] ** p)
    return np.column_stack(cols)


def regress_conditional_expectation(
    conditioning: np.ndarray,
    target: np.ndarray,
    degree: int = 4,
) -> np.ndarray:
    """Least-squares estimate of ``E[target | sigma(conditioning)]``.

    Parameters
    ----------
    conditioning:
        Array of shape ``(n_paths,)`` or ``(n_paths, dim)`` with the
        F_s-measurable conditioning variables. If it has zero columns or is
        constant (trivial sigma-field, e.g. s = 0), the conditional
        expectation reduces to the unconditional sample mean.
    target:
        Array of shape ``(n_paths,)`` (or ``(n_paths, k)``) with the
        quantity whose conditional expectation is estimated.
    degree:
        Polynomial degree of the regression basis.

    Returns
    -------
    np.ndarray
        Fitted values of the same leading shape as ``target``.
    """
    target = np.asarray(target, dtype=float)
    conditioning = np.asarray(conditioning, dtype=float)
    if conditioning.ndim == 1:
        conditioning = conditioning[:, None]
    if conditioning.shape[1] == 0 or np.allclose(conditioning.std(axis=0), 0.0):
        mean = target.mean(axis=0)
        return np.broadcast_to(mean, target.shape).copy()

    design = polynomial_basis(conditioning, degree)
    flat_target = target.reshape(target.shape[0], -1)
    coeffs, *_ = np.linalg.lstsq(design, flat_target, rcond=None)
    fitted = design @ coeffs
    return fitted.reshape(target.shape)
