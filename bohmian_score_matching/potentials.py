"""Potentials used in the PRL benchmarks.

Two systems:

* ``morse-chain``: ``d`` coupled Morse oscillators,
  V = sum_i D (1 - exp(-beta x_i))^2 + lam sum_i x_i x_{i+1},
  initial state a product of harmonic-ground-state Gaussians with the first
  mode displaced by q0.
* ``double-well``: 1D symmetric quartic,
  V(x) = D (x^2/a^2 - 1)^2,
  initial state a Gaussian centred on the barrier top.

Each factory returns a dict with the potential function, initial Gaussian
moments, and physical constants.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def morse_chain(d=4, D=12.5, beta=0.2, lam=0.3, q0=1.0, sigma0=None,
                hbar=1.0, mass=1.0):
    """Coupled Morse chain (PRL d=4 benchmark).

    Args:
        d: number of oscillators.
        D, beta: on-site Morse parameters.
        lam: nearest-neighbour bilinear coupling.
        q0: initial displacement of the first mode.
        sigma0: initial Gaussian width per mode.  Defaults to the harmonic
            ground-state width sqrt(hbar / (2 m omega0)) with
            omega0 = beta sqrt(2 D).
    """
    omega0 = beta * jnp.sqrt(2.0 * D)
    if sigma0 is None:
        sigma0 = jnp.sqrt(hbar / (2.0 * mass * omega0))

    def V_fn(r):
        on_site = jnp.sum(D * (1.0 - jnp.exp(-beta * r)) ** 2)
        coupling = lam * jnp.sum(r[:-1] * r[1:])
        return on_site + coupling

    r0_mean = jnp.zeros(d).at[0].set(q0)
    r0_cov = jnp.eye(d) * sigma0 ** 2
    v0 = jnp.zeros(d)

    return {
        'V_fn': V_fn,
        'r0_mean': r0_mean,
        'r0_cov': r0_cov,
        'v0': v0,
        'hbar': hbar,
        'mass': mass,
        'label': f'Morse chain (d={d}, D={D}, beta={beta}, lam={lam})',
        'name': 'morse-chain',
        'd': d, 'D': D, 'beta': beta, 'lam': lam,
    }


def double_well(a=2.0, D=1.0, sigma0=0.5, hbar=1.0, mass=1.0):
    """1D symmetric double well V(x) = D (x^2/a^2 - 1)^2.

    Initial Gaussian sits on the barrier at x = 0 with width sigma0.
    """
    def V_fn(r):
        return D * (r[0] ** 2 / a ** 2 - 1.0) ** 2

    r0_mean = jnp.array([0.0])
    r0_cov = jnp.array([[sigma0 ** 2]])
    v0 = jnp.array([0.0])

    return {
        'V_fn': V_fn,
        'r0_mean': r0_mean,
        'r0_cov': r0_cov,
        'v0': v0,
        'hbar': hbar,
        'mass': mass,
        'label': f'Double well (a={a}, D={D}, sigma0={sigma0})',
        'name': 'double-well',
        'a': a, 'D': D, 'sigma0': sigma0,
    }


def get_system(name, **kwargs):
    """Dispatch to the named system."""
    registry = {
        'morse-chain': morse_chain,
        'double-well': double_well,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown system '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
