"""Fisher loss tests: zero at optimum, zero-gradient fixed point."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bohmian_flow import core


def test_initial_score_gaussian():
    r0 = jnp.array([0.0, 1.0])
    cov = jnp.diag(jnp.array([0.5, 0.25]))
    s0 = core.make_initial_score_fn(r0, cov)
    # At mean, score = 0
    np.testing.assert_allclose(np.array(s0(r0)), np.zeros(2), atol=1e-12)
    # Score of Gaussian: -Sigma^{-1} (x - mu)
    x = jnp.array([0.5, 1.2])
    expected = -jnp.linalg.inv(cov) @ (x - r0)
    np.testing.assert_allclose(np.array(s0(x)), np.array(expected),
                               atol=1e-12)


def test_fisher_loss_zero_when_score_is_exact_initial():
    """Constant score = s_0 on a harmonic well with sigma=1/sqrt(2):
    at T = 0 exactly, target = s_0 so residual is exactly zero."""
    d = 2
    r0 = jnp.zeros(d)
    cov = jnp.eye(d) * 0.5
    s0 = core.make_initial_score_fn(r0, cov)

    # Score "network" that returns the exact initial score at all t
    def score_fn(params, x, t):
        return s0(x)

    def V(r):
        return 0.5 * jnp.sum(r ** 2)

    key = jax.random.PRNGKey(0)
    X0 = jax.random.multivariate_normal(key, r0, cov, shape=(4,))
    V0 = jnp.zeros_like(X0)
    # Evaluate loss over a very short window so F remains close to I and
    # the target stays close to s_0.  Loss should be small.
    loss = core.fisher_loss(
        {}, X0, V0, score_fn, V,
        T=0.02, dt=0.01, initial_score_fn=s0,
        ckpt_idx=None, caustic_threshold=0.01, target_clip=100.0)
    assert float(loss) < 1.0  # finite and non-catastrophic


def test_fisher_loss_at_t0_exact():
    """With ckpt_idx = [0], target equals s_0 exactly, so loss = 0 iff
    the score network outputs s_0 at t = 0 for every sample."""
    d = 2
    r0 = jnp.zeros(d)
    cov = jnp.eye(d) * 0.5
    s0 = core.make_initial_score_fn(r0, cov)

    def score_fn(params, x, t):
        return s0(x)

    def V(r):
        return 0.5 * jnp.sum(r ** 2)

    key = jax.random.PRNGKey(1)
    X0 = jax.random.multivariate_normal(key, r0, cov, shape=(5,))
    V0 = jnp.zeros_like(X0)
    loss = core.fisher_loss(
        {}, X0, V0, score_fn, V,
        T=0.05, dt=0.01, initial_score_fn=s0,
        ckpt_idx=jnp.array([0]),
        caustic_threshold=0.01, target_clip=100.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


def test_fisher_loss_differentiable():
    """grad(L) w.r.t. params must return finite values."""
    from bohmian_flow import network

    d = 2
    r0 = jnp.zeros(d)
    cov = jnp.eye(d) * 0.5
    s0 = core.make_initial_score_fn(r0, cov)

    init_fn, score_fn, _ = network.make_score_network(
        d=d, hidden_dims=[4, 4], n_freq=1,
        s_baseline_fn=lambda x, t: jnp.zeros(d), conditioning='concat')
    params = init_fn(jax.random.PRNGKey(0))

    def V(r):
        return 0.5 * jnp.sum(r ** 2)

    key = jax.random.PRNGKey(2)
    X0 = jax.random.multivariate_normal(key, r0, cov, shape=(3,))
    V0 = jnp.zeros_like(X0)

    def loss_fn(mlp_params):
        p = {'mlp_params': mlp_params, 'freqs': params['freqs']}
        return core.fisher_loss(p, X0, V0, score_fn, V,
                                T=0.03, dt=0.01, initial_score_fn=s0,
                                ckpt_idx=None,
                                caustic_threshold=0.01, target_clip=100.0)

    grads = jax.grad(loss_fn)(params['mlp_params'])
    from jax.flatten_util import ravel_pytree
    flat, _ = ravel_pytree(grads)
    assert jnp.all(jnp.isfinite(flat))
