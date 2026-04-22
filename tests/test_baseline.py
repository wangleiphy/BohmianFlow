"""Gaussian baseline tests: Harmonic approximation + Ermakov time evolution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bohmian_flow import baseline


def test_harmonic_approximation_identifies_eigenmodes():
    """Hessian of V = 1/2 diag(omega_i^2) x^2 returns the same omegas."""
    omegas = jnp.array([1.0, 2.0, 3.0])

    def V(r):
        return 0.5 * jnp.sum(omegas ** 2 * r ** 2)

    r_eq, omega, U = baseline.harmonic_approximation(V, d=3)
    # Up to eigenvalue ordering
    np.testing.assert_allclose(np.sort(np.array(omega)),
                               np.sort(np.array(omegas)), atol=1e-6)
    np.testing.assert_allclose(np.array(r_eq), np.zeros(3), atol=1e-5)


def test_baseline_score_initial_state():
    """At t=0, s_base(r0_mean) = 0 (centered Gaussian score has zero mean)."""
    def V(r):
        return 0.5 * jnp.sum(r ** 2)

    r0 = jnp.array([1.0, 0.0])
    cov = jnp.eye(2) * 0.5
    p0 = jnp.zeros(2)
    s_base = baseline.make_baseline_score(V, r0, cov, p0)
    s = s_base(r0, 0.0)
    np.testing.assert_allclose(np.array(s), np.zeros(2), atol=1e-8)


def test_baseline_score_gradient_field():
    """s_base is a gradient field => Jacobian is symmetric."""
    def V(r):
        return 0.5 * (r[0] ** 2 + 2.0 * r[1] ** 2)

    r0 = jnp.array([0.3, 0.0])
    cov = jnp.diag(jnp.array([0.5, 0.25]))
    p0 = jnp.array([0.2, 0.0])
    s_base = baseline.make_baseline_score(V, r0, cov, p0)
    J = jax.jacobian(s_base, argnums=0)(jnp.array([0.4, 0.1]), 0.3)
    np.testing.assert_allclose(np.array(J), np.array(J).T, atol=1e-8)
