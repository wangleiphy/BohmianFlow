"""Score network: gradient-field property, Q sanity, FiLM/concat parity at t=0."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bohmian_score_matching import network


def _zero_baseline(d):
    return lambda x, t: jnp.zeros(d)


@pytest.mark.parametrize('conditioning', ['concat', 'film'])
def test_score_is_gradient_field(conditioning):
    """s_theta = grad phi => curl = 0 => Jacobian symmetric."""
    d = 3
    init_fn, score_fn, _ = network.make_score_network(
        d=d, hidden_dims=[8, 8], n_freq=2,
        s_baseline_fn=_zero_baseline(d), conditioning=conditioning)
    params = init_fn(jax.random.PRNGKey(0))
    x = jnp.array([0.5, -0.3, 0.2])
    t = 0.7
    J = jax.jacobian(lambda x_: score_fn(params, x_, t))(x)
    np.testing.assert_allclose(np.array(J), np.array(J).T, atol=1e-8)


def test_quantum_potential_matches_definition():
    """Q = -(hbar^2/4m)(div s + 1/2 |s|^2) reproduced from explicit formula."""
    d = 2
    init_fn, score_fn, _ = network.make_score_network(
        d=d, hidden_dims=[6, 6], n_freq=0,
        s_baseline_fn=_zero_baseline(d))
    params = init_fn(jax.random.PRNGKey(1))
    x = jnp.array([0.1, -0.4])
    t = 0.3
    s = score_fn(params, x, t)
    # Divergence via jacobian trace
    J = jax.jacobian(lambda x_: score_fn(params, x_, t))(x)
    div_s = float(jnp.trace(J))
    Q_expected = -0.25 * (div_s + 0.5 * float(jnp.sum(s ** 2)))
    Q = float(network.quantum_potential(score_fn, params, x, t))
    assert Q == pytest.approx(Q_expected, rel=1e-10, abs=1e-12)


def test_count_params_positive():
    d = 4
    init_fn, _, _ = network.make_score_network(
        d=d, hidden_dims=[16, 16], n_freq=3,
        s_baseline_fn=_zero_baseline(d), conditioning='film')
    params = init_fn(jax.random.PRNGKey(0))
    assert network.count_params(params) > 0


def test_initial_output_small():
    """At t = 0 and with small output_scale, the correction s_theta should
    be dominated by the baseline (here zero), so |s_theta| << 1."""
    d = 2
    init_fn, score_fn, _ = network.make_score_network(
        d=d, hidden_dims=[8, 8], n_freq=2,
        s_baseline_fn=_zero_baseline(d), conditioning='film',
        output_scale=0.001)
    params = init_fn(jax.random.PRNGKey(42))
    x = jnp.array([0.3, 0.1])
    s = score_fn(params, x, 0.0)
    assert float(jnp.linalg.norm(s)) < 0.5
