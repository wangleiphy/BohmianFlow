"""Trajectory integrator: energy conservation + F identity at t=0."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bohmian_flow import trajectory


def _zero_score(params, x, t):
    return jnp.zeros_like(x)


def test_F_initial_is_identity():
    """F(0) = I, G(0) = 0 by the variational equations."""
    x0 = jnp.array([0.1, 0.2])
    v0 = jnp.array([0.0, 0.0])

    def V(r):
        return 0.5 * jnp.sum(r ** 2)

    x_traj, v_traj, F_traj, _ = trajectory.propagate_with_F(
        x0, v0, {}, _zero_score, V, T=0.1, dt=0.01)
    np.testing.assert_allclose(np.array(F_traj[0]), np.eye(2), atol=1e-12)


def test_harmonic_F_matches_analytic():
    """In a 1D harmonic oscillator with zero score, F(t) = cos(omega t)."""
    x0 = jnp.array([0.5])
    v0 = jnp.array([0.0])

    def V(r):
        return 0.5 * r[0] ** 2

    T, dt = 1.0, 0.002
    x_traj, _, F_traj, t_traj = trajectory.propagate_with_F(
        x0, v0, {}, _zero_score, V, T, dt)
    F_expected = np.cos(np.array(t_traj))
    np.testing.assert_allclose(np.array(F_traj[:, 0, 0]), F_expected,
                               atol=5e-4)


def test_leapfrog_energy_drift_small():
    """Leapfrog: 1D harmonic, energy conserved to O(dt^2)."""
    x0 = jnp.array([0.5])
    v0 = jnp.array([0.2])

    def V(r):
        return 0.5 * r[0] ** 2

    x_traj, v_traj, _, _ = trajectory.propagate_with_F(
        x0, v0, {}, _zero_score, V, T=10.0, dt=0.01)
    E = 0.5 * v_traj[:, 0] ** 2 + 0.5 * x_traj[:, 0] ** 2
    drift = float(jnp.max(jnp.abs(E - E[0])))
    assert drift < 1e-3


def test_propagate_and_propagate_with_F_agree():
    """propagate (no F) and propagate_with_F give identical x, v trajectories."""
    x0 = jnp.array([0.3, -0.1])
    v0 = jnp.array([0.1, 0.0])

    def V(r):
        return 0.5 * jnp.sum(r ** 2)

    x1, v1, _ = trajectory.propagate(
        x0, v0, {}, _zero_score, V, T=0.3, dt=0.01)
    x2, v2, _, _ = trajectory.propagate_with_F(
        x0, v0, {}, _zero_score, V, T=0.3, dt=0.01)
    np.testing.assert_allclose(np.array(x1), np.array(x2), atol=1e-12)
    np.testing.assert_allclose(np.array(v1), np.array(v2), atol=1e-12)
