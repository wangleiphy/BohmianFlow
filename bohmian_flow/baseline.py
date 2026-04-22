"""Gaussian baseline score from the harmonic approximation.

The baseline
    s_base(x, t) = -Sigma(t)^{-1} (x - x_c(t))
is the exact score of the Gaussian state that the system reduces to when V is
replaced by its harmonic expansion around the equilibrium r_eq.  Adding this
as an untrained background to the score network gives the training a warm
start: the network only has to learn the anharmonic correction rather than
the dominant harmonic-oscillator structure.

Parameters of the baseline (normal-mode frequencies, initial widths) are
computed once at setup from V, r0_mean, r0_cov, p0_mean via BFGS + eigh.
Ermakov's equation supplies the analytic time evolution of the widths.
"""

import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def harmonic_approximation(V_fn, d):
    """Find equilibrium of V, diagonalise the Hessian.

    Returns:
        r_eq: (d,) equilibrium position.
        omega: (d,) normal-mode frequencies.
        U: (d, d) eigenvectors of Hess(V) at r_eq.
    """
    r0 = jnp.zeros(d)
    result = minimize(lambda r: float(V_fn(jnp.array(r))), r0, method='BFGS')
    if not result.success:
        raise RuntimeError(f"Failed to find equilibrium: {result.message}")
    r_eq = jnp.array(result.x)

    H = jax.hessian(V_fn)(r_eq)
    eigenvalues, U = jnp.linalg.eigh(H)
    if jnp.any(eigenvalues < -1e-10):
        raise ValueError(f"Negative Hessian eigenvalues: {eigenvalues}")
    omega = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    return r_eq, omega, U


def _ermakov(omega, sigma0, t):
    """Analytic sigma^2(t) for a Gaussian state in a harmonic well.

    sigma^2(t) = A + B cos(2 omega t) with
    A = 1/2 (sigma0^2 + sigma_eq^4 / sigma0^2),
    B = 1/2 (sigma0^2 - sigma_eq^4 / sigma0^2),
    sigma_eq = 1/sqrt(2 omega) (ground-state width).
    """
    sigma_eq2 = 1.0 / (2.0 * omega)
    A = 0.5 * (sigma0 ** 2 + sigma_eq2 ** 2 / sigma0 ** 2)
    B = 0.5 * (sigma0 ** 2 - sigma_eq2 ** 2 / sigma0 ** 2)
    sigma2 = A + B * jnp.cos(2.0 * omega * t)
    return sigma2


def make_baseline_score(V_fn, r0_mean, r0_cov, p0_mean):
    """Build the Gaussian baseline score s_base(x, t) from the harmonic approximation.

    Returns a function ``s_base(x, t) -> (d,)`` suitable to pass as the
    baseline of :func:`make_score_network`.
    """
    d = r0_mean.shape[0]
    r_eq, omega, U = harmonic_approximation(V_fn, d)

    q0_mean = U.T @ (r0_mean - r_eq)
    p_q0_mean = U.T @ p0_mean
    Sigma_q0 = U.T @ r0_cov @ U
    sigma0_modes = jnp.sqrt(jnp.diag(Sigma_q0))

    def s_baseline_fn(x, t):
        # Classical centroid of each normal mode
        q_c = q0_mean * jnp.cos(omega * t) + (p_q0_mean / omega) * jnp.sin(omega * t)
        x_c = r_eq + U @ q_c
        # Current width of each mode
        sigma2 = _ermakov(omega, sigma0_modes, t)
        # Score in normal-mode frame, then rotate back
        dq = U.T @ (x - x_c)
        return U @ (-dq / sigma2)

    return s_baseline_fn
