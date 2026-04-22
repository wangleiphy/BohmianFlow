"""Split-operator FFT reference solver.

Strang splitting of the TDSE kinetic-potential factorisation,

    psi(t + dt) = exp(-i V dt / 2 hbar) exp(-i T dt / hbar) exp(-i V dt / 2 hbar) psi(t),

implemented in NumPy on a uniform tensor grid.  Supports arbitrary dimension
d; memory scales as O(N^d) for N grid points per axis (d <= 4 at N = 64
fits on a workstation).

Helpers:
    gaussian_wavepacket_nd: normalised Gaussian initial wave function.
    compute_observables_nd: <x_i> and sigma_i from |psi|^2.
    split_operator_nd:      main time-integrator, returns observable history
                             plus the final wave function and E_exact.

A convenience 1D Bohmian-velocity integrator
:func:`bohmian_trajectories_1d` is also provided for the double-well
figure: it co-integrates particle positions x_i(t) with the same FFT solver
using the exact Bohmian velocity v = Im(psi'/psi).
"""

import numpy as np


def gaussian_wavepacket_nd(grids, r0, sigma, p0=None, hbar=1.0):
    """Normalised d-dimensional Gaussian on a tensor grid.

    Args:
        grids: list of d 1D coordinate arrays.
        r0: (d,) centre.
        sigma: (d,) per-axis width (standard deviation of |psi|^2 is sigma).
        p0: (d,) initial momentum; default zero.
        hbar: reduced Planck constant.

    Returns:
        psi: complex array of shape (N_0, ..., N_{d-1}), normalised so
        int |psi|^2 dV = 1 on the given grid.
    """
    d = len(grids)
    if p0 is None:
        p0 = np.zeros(d)

    mesh = np.meshgrid(*grids, indexing='ij')
    dV = np.prod([g[1] - g[0] for g in grids])

    exponent = sum(-(mesh[i] - r0[i]) ** 2 / (4.0 * sigma[i] ** 2)
                   for i in range(d))
    psi = np.exp(exponent)
    phase = sum(p0[i] * mesh[i] for i in range(d)) / hbar
    psi = psi * np.exp(1j * phase)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dV)
    return psi / norm


def compute_observables_nd(psi, grids):
    """Per-axis <x_i> and sigma_i from |psi|^2."""
    d = len(grids)
    rho = np.abs(psi) ** 2
    dV = np.prod([g[1] - g[0] for g in grids])
    norm = np.sum(rho) * dV
    mesh = np.meshgrid(*grids, indexing='ij')
    means = np.array([np.sum(mesh[i] * rho) * dV / norm for i in range(d)])
    sigmas = np.array([
        np.sqrt(max(np.sum(mesh[i] ** 2 * rho) * dV / norm - means[i] ** 2, 0.0))
        for i in range(d)
    ])
    return means, sigmas


def split_operator_nd(V_of_mesh, grids, psi0, T, dt, hbar=1.0, mass=1.0,
                     checkpoint_every=1, save_psi_at=None, verbose=False):
    """Strang split-operator integrator of the TDSE.

    Args:
        V_of_mesh: callable ``V(*mesh) -> array``, where
            ``mesh = np.meshgrid(*grids, indexing='ij')``.
        grids: list of d 1D coordinate arrays.
        psi0: initial wave function, shape ``(N_0, ..., N_{d-1})``.
        T, dt: total time and timestep.
        hbar, mass: physical constants.
        checkpoint_every: save means/sigmas every N steps.
        save_psi_at: optional list of times at which to keep a full copy of
            psi (snapshots).
        verbose: print norm every N_steps / 5 steps.

    Returns:
        dict with keys ``psi_final``, ``psi_snapshots`` (list of
        ``(psi, t)``), ``means``, ``sigmas``, ``t``, ``E_exact``.
    """
    d = len(grids)
    n_steps = int(T / dt)

    mesh = np.meshgrid(*grids, indexing='ij')
    dV = np.prod([g[1] - g[0] for g in grids])
    V = V_of_mesh(*mesh)

    k_grids = [2 * np.pi * np.fft.fftfreq(len(g), d=g[1] - g[0]) for g in grids]
    k_mesh = np.meshgrid(*k_grids, indexing='ij')
    K2 = sum(ki ** 2 for ki in k_mesh)

    expV_half = np.exp(-0.5j * V * dt / hbar)
    expT = np.exp(-1j * hbar * K2 / (2.0 * mass) * dt)

    psi_k = np.fft.fftn(psi0)
    n_total = psi0.size
    KE0 = np.real(np.sum(np.conj(psi_k) * (hbar ** 2 * K2 / (2 * mass)) * psi_k)
                  ) * dV / n_total
    PE0 = np.sum(V * np.abs(psi0) ** 2) * dV
    E_exact = float(KE0 + PE0)

    ckpt_steps = list(range(0, n_steps + 1, checkpoint_every))
    if ckpt_steps[-1] != n_steps:
        ckpt_steps.append(n_steps)
    K_ck = len(ckpt_steps)
    means = np.zeros((K_ck, d))
    sigmas = np.zeros((K_ck, d))
    t_arr = np.zeros(K_ck)

    psi = psi0.copy()
    ci = 0
    if ckpt_steps[ci] == 0:
        m, s = compute_observables_nd(psi, grids)
        means[ci], sigmas[ci], t_arr[ci] = m, s, 0.0
        ci += 1

    snap_steps = {}
    if save_psi_at is not None:
        for t_val in save_psi_at:
            step = int(round(t_val / dt))
            snap_steps.setdefault(step, t_val)

    snapshots = []
    if 0 in snap_steps:
        snapshots.append((psi.copy(), snap_steps[0]))

    for step in range(1, n_steps + 1):
        psi = expV_half * psi
        psi = np.fft.ifftn(expT * np.fft.fftn(psi))
        psi = expV_half * psi

        if ci < K_ck and step == ckpt_steps[ci]:
            m, s = compute_observables_nd(psi, grids)
            means[ci], sigmas[ci], t_arr[ci] = m, s, step * dt
            ci += 1

        if step in snap_steps:
            snapshots.append((psi.copy(), snap_steps[step]))

        if verbose and step % max(1, n_steps // 5) == 0:
            norm = np.sum(np.abs(psi) ** 2) * dV
            print(f"  FFT step {step}/{n_steps}, norm={norm:.10f}")

    return {
        'psi_final': psi,
        'psi_snapshots': snapshots,
        'means': means, 'sigmas': sigmas, 't': t_arr,
        'grids': grids, 'k_grids': k_grids,
        'E_exact': E_exact,
    }


# --- 1D Bohmian-velocity integrator ----------------------------------------

def bohmian_trajectories_1d(x_grid, psi0, V, T, dt, x_particles,
                            hbar=1.0, mass=1.0, save_every=1):
    """Co-integrate a 1D FFT evolution with exact Bohmian trajectories.

    The Bohmian velocity on the grid is v(x) = (hbar/m) Im(psi'(x) / psi(x))
    with psi' computed spectrally.  Particle velocities are linearly
    interpolated from the grid at each step.

    Returns:
        rho_arr: (n_save, N) |psi|^2 at the save times.
        traj_x: (n_save, n_particles) particle positions.
        t_arr: (n_save,) save times.
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    expV_half = np.exp(-0.5j * V * dt / hbar)
    expT = np.exp(-0.5j * hbar * k ** 2 / mass * dt)

    n_steps = int(T / dt)
    save_steps = list(range(0, n_steps + 1, save_every))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    n_save = len(save_steps)

    psi = psi0.copy()
    xp = np.asarray(x_particles, dtype=float).copy()

    rho_arr = np.zeros((n_save, N))
    traj_x = np.zeros((n_save, xp.size))
    t_arr = np.zeros(n_save)

    si = 0
    if save_steps[0] == 0:
        rho_arr[si] = np.abs(psi) ** 2
        traj_x[si] = xp.copy()
        t_arr[si] = 0.0
        si += 1

    for step in range(1, n_steps + 1):
        # Bohmian velocity from current psi (before the FFT update).
        psi_k = np.fft.fft(psi)
        dpsi_dx = np.fft.ifft(1j * k * psi_k)
        safe_psi = np.where(np.abs(psi) > 1e-30, psi, 1e-30)
        v_grid = np.where(np.abs(psi) > 1e-30,
                          hbar / mass * np.imag(dpsi_dx / safe_psi), 0.0)
        v_p = np.interp(xp, x_grid, v_grid)
        xp = xp + v_p * dt

        # Update psi via split-operator.
        psi = expV_half * psi
        psi = np.fft.ifft(expT * np.fft.fft(psi))
        psi = expV_half * psi

        if si < n_save and step == save_steps[si]:
            rho_arr[si] = np.abs(psi) ** 2
            traj_x[si] = xp.copy()
            t_arr[si] = step * dt
            si += 1

    return rho_arr, traj_x, t_arr
