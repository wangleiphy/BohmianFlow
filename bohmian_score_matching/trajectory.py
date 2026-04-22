"""Single-particle leapfrog with deformation-gradient co-integration.

Integrates the Bohmian equations of motion

    dx/dt = v,   m dv/dt = -grad(V + Q),

together with the variational equations

    dF/dt = G,   m dG/dt = -Hess(V + Q) . F,

using a symplectic leapfrog (kick-drift-kick) scheme inside ``lax.scan``.
Each single-particle propagation is JIT-compatible and differentiable for
BPTT through the trajectory, thanks to ``jax.checkpoint`` on the scan body
(memory traded for recomputation).

``Hess(V+Q) . F`` is evaluated as d JVPs of the acceleration field, which
avoids materialising the full d x d Hessian and keeps the per-step cost at
O(d^2).

Outer ``vmap`` over the particle ensemble is applied by the caller
(e.g. :func:`fisher_loss`).
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_acceleration_fn(V_fn, score_fn, params, hbar=1.0, mass=1.0):
    """Build the Bohmian acceleration a(x, t) = -(grad V + grad Q) / m.

    Q is computed from the score network s_theta through

        Q = -(hbar^2 / 4 m) (div s + 1/2 |s|^2),

    and grad Q is obtained by ``jax.grad`` of Q. This makes the acceleration
    depend on first, second, and third spatial derivatives of the scalar
    potential phi_theta behind s_theta = grad phi_theta.
    """
    from bohmian_score_matching.network import quantum_potential

    def accel_fn(x, t):
        F_cl = -jax.grad(V_fn)(x)
        F_Q = -jax.grad(
            lambda x_: quantum_potential(score_fn, params, x_, t, hbar, mass)
        )(x)
        return (F_cl + F_Q) / mass

    return accel_fn


def propagate_with_F(x0, v0, params, score_fn, V_fn, T, dt,
                     hbar=1.0, mass=1.0):
    """Propagate a single particle, co-integrating (F, G) alongside (x, v).

    Returns:
        x_traj: (n_steps+1, d) positions.
        v_traj: (n_steps+1, d) velocities.
        F_traj: (n_steps+1, d, d) deformation gradients F = dx(t)/dx(0).
        t_traj: (n_steps+1,) times.
    """
    d = x0.shape[0]
    n_steps = int(T / dt)

    accel_fn = make_acceleration_fn(V_fn, score_fn, params, hbar, mass)

    def compute_HF(x, t, F):
        # (Hess a) @ F = d/dx[a(x)] . F, evaluated as d JVPs
        def hf_column(col):
            _, out = jax.jvp(lambda x_: accel_fn(x_, t), (x,), (col,))
            return out
        return jax.vmap(hf_column)(F.T).T

    F0 = jnp.eye(d)
    G0 = jnp.zeros((d, d))
    a0 = accel_fn(x0, 0.0)
    HF0 = compute_HF(x0, 0.0, F0)

    init_carry = (x0, v0, F0, G0, a0, HF0, 0)

    @jax.checkpoint
    def leapfrog_step(carry, _):
        x, v, F, G, a_old, HF_old, step_idx = carry

        v_half = v + (dt / 2) * a_old
        G_half = G + (dt / 2) * HF_old

        x_new = x + dt * v_half
        F_new = F + dt * G_half

        t_new = (step_idx + 1) * dt
        a_new = accel_fn(x_new, t_new)
        HF_new = compute_HF(x_new, t_new, F_new)

        v_new = v_half + (dt / 2) * a_new
        G_new = G_half + (dt / 2) * HF_new

        new_carry = (x_new, v_new, F_new, G_new, a_new, HF_new, step_idx + 1)
        return new_carry, (x_new, v_new, F_new)

    _, (x_all, v_all, F_all) = jax.lax.scan(
        leapfrog_step, init_carry, None, length=n_steps
    )

    x_traj = jnp.concatenate([x0[None], x_all], axis=0)
    v_traj = jnp.concatenate([v0[None], v_all], axis=0)
    F_traj = jnp.concatenate([F0[None], F_all], axis=0)
    t_traj = jnp.arange(n_steps + 1) * dt
    return x_traj, v_traj, F_traj, t_traj


def propagate(x0, v0, params, score_fn, V_fn, T, dt, hbar=1.0, mass=1.0):
    """Propagate a single particle without carrying F.

    Convenience wrapper used for inference / plotting, where the deformation
    gradient is not needed and the reduced carry cuts memory.
    """
    n_steps = int(T / dt)
    accel_fn = make_acceleration_fn(V_fn, score_fn, params, hbar, mass)
    a0 = accel_fn(x0, 0.0)
    init_carry = (x0, v0, a0, 0)

    @jax.checkpoint
    def leapfrog_step(carry, _):
        x, v, a_old, step_idx = carry
        v_half = v + (dt / 2) * a_old
        x_new = x + dt * v_half
        t_new = (step_idx + 1) * dt
        a_new = accel_fn(x_new, t_new)
        v_new = v_half + (dt / 2) * a_new
        return (x_new, v_new, a_new, step_idx + 1), (x_new, v_new)

    _, (x_all, v_all) = jax.lax.scan(
        leapfrog_step, init_carry, None, length=n_steps
    )
    x_traj = jnp.concatenate([x0[None], x_all], axis=0)
    v_traj = jnp.concatenate([v0[None], v_all], axis=0)
    t_traj = jnp.arange(n_steps + 1) * dt
    return x_traj, v_traj, t_traj
