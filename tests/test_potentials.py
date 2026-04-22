"""Potentials: shape/initial conditions and a few analytic cross-checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bohmian_score_matching import potentials


def test_morse_chain_defaults():
    sys = potentials.morse_chain(d=4, q0=1.0)
    assert sys['r0_mean'].shape == (4,)
    assert sys['r0_cov'].shape == (4, 4)
    assert float(sys['r0_mean'][0]) == pytest.approx(1.0)
    # Other coords at origin
    for i in range(1, 4):
        assert float(sys['r0_mean'][i]) == pytest.approx(0.0)


def test_morse_chain_equilibrium_energy():
    """V(0) should be 0 when all q_i = 0 (Morse minimum at q = 0)."""
    sys = potentials.morse_chain(d=3, q0=0.0)
    r = jnp.zeros(3)
    assert float(sys['V_fn'](r)) == pytest.approx(0.0, abs=1e-12)


def test_morse_chain_coupling_sign():
    """Turning on lam > 0 with positive q_0 q_1 product raises V."""
    sys = potentials.morse_chain(d=2, q0=0.0, lam=0.3)
    r = jnp.array([0.5, 0.5])
    # sum of on-site + lam*0.25
    V_on_site = 2 * 12.5 * (1 - np.exp(-0.2 * 0.5)) ** 2
    assert float(sys['V_fn'](r)) == pytest.approx(V_on_site + 0.3 * 0.25,
                                                  rel=1e-10)


def test_double_well_minima():
    """Double-well: minima at x = +-a with V = 0."""
    sys = potentials.double_well(a=2.0, D=1.0)
    for x in (2.0, -2.0):
        V = sys['V_fn'](jnp.array([x]))
        assert float(V) == pytest.approx(0.0, abs=1e-12)
    # Barrier at 0 with height D
    assert float(sys['V_fn'](jnp.array([0.0]))) == pytest.approx(1.0, abs=1e-12)


def test_get_system_dispatch():
    sys = potentials.get_system('morse-chain', d=2)
    assert sys['d'] == 2
    sys = potentials.get_system('double-well')
    assert sys['name'] == 'double-well'
    with pytest.raises(ValueError):
        potentials.get_system('nope')
