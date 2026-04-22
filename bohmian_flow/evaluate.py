"""Inference helpers: sample initial conditions, propagate, collect moments."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def sample_initial_conditions(key, M, r0_mean, r0_cov, v0):
    """Sample M particles from the initial Gaussian; shared initial velocity."""
    X0 = jax.random.multivariate_normal(key, r0_mean, r0_cov, shape=(M,))
    V0 = jnp.tile(v0, (M, 1))
    return X0, V0


def evaluate_trajectories(score_fn, params, V_fn, X0, V0, T, dt,
                          n_checkpoints, hbar=1.0, mass=1.0):
    """Propagate ensemble with the trained score, compute per-mode means, widths, energies.

    Args:
        score_fn, params, V_fn: as during training.
        X0, V0: (M, d) test initial conditions.
        T, dt: time horizon and leapfrog step.
        n_checkpoints: number of output time points evenly spaced in [0, T].

    Returns:
        dict with ``mean_x: (K, d)``, ``sigma: (K, d)``, ``energies: (M, K)``,
        ``t: (K,)``.
    """
    from bohmian_flow.trajectory import propagate_with_F
    from bohmian_flow.network import quantum_potential_batch

    n_steps = int(T / dt)

    def single(x0, v0):
        x_traj, v_traj, _F_traj, t_traj = propagate_with_F(
            x0, v0, params, score_fn, V_fn, T, dt, hbar, mass)
        return x_traj, v_traj, t_traj

    x_all, v_all, t_traj = jax.vmap(single)(X0, V0)
    # x_all: (M, n_steps+1, d)

    ck = jnp.linspace(0, n_steps, n_checkpoints).astype(int)
    x_ck = x_all[:, ck, :]
    v_ck = v_all[:, ck, :]
    t_ck = t_traj[0, ck]

    mean_x = jnp.mean(x_ck, axis=0)
    sigma = jnp.std(x_ck, axis=0)

    def _energy_at_t(X, V, t_val):
        Q = quantum_potential_batch(score_fn, params, X, t_val, hbar, mass)
        KE = 0.5 * mass * jnp.sum(V ** 2, axis=1)
        PE = jax.vmap(V_fn)(X)
        return KE + PE + Q

    energies = jax.vmap(_energy_at_t)(
        x_ck.transpose(1, 0, 2),
        v_ck.transpose(1, 0, 2),
        t_ck,
    ).T  # (M, K)

    return {
        'mean_x': mean_x,
        'sigma': sigma,
        'energies': energies,
        't': t_ck,
    }
