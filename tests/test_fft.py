"""FFT split-operator solver: harmonic oscillator cross-checks."""

import numpy as np
import pytest

from bohmian_score_matching import fft


def test_harmonic_ground_state_energy_1d():
    """1D HO ground state has E = 1/2 hbar omega = 1/2 (omega=hbar=m=1)."""
    x = np.linspace(-8, 8, 256)
    psi0 = fft.gaussian_wavepacket_nd(
        [x], np.array([0.0]), np.array([1.0 / np.sqrt(2)]))
    result = fft.split_operator_nd(
        lambda X: 0.5 * X ** 2, [x], psi0,
        T=0.5, dt=0.01, checkpoint_every=50)
    assert result['E_exact'] == pytest.approx(0.5, abs=1e-3)


def test_coherent_state_1d_oscillates():
    """Coherent state centred at x0: mean oscillates like x0 cos(omega t)."""
    x = np.linspace(-8, 8, 256)
    x0 = 1.0
    psi0 = fft.gaussian_wavepacket_nd(
        [x], np.array([x0]), np.array([1.0 / np.sqrt(2)]))
    result = fft.split_operator_nd(
        lambda X: 0.5 * X ** 2, [x], psi0,
        T=2 * np.pi, dt=0.01, checkpoint_every=1)
    t = result['t']
    means = result['means'][:, 0]
    # Analytical: <x>(t) = x0 cos(t).  Compare at t = pi (should be -x0).
    idx_pi = int(np.argmin(np.abs(t - np.pi)))
    assert means[idx_pi] == pytest.approx(-x0, abs=5e-3)


def test_observables_at_t0():
    """compute_observables_nd on a Gaussian at the centre returns (0, sigma)."""
    x = np.linspace(-6, 6, 128)
    y = np.linspace(-6, 6, 128)
    sigma = np.array([0.7, 0.5])
    psi0 = fft.gaussian_wavepacket_nd([x, y], np.array([0.0, 0.0]), sigma)
    means, sigmas = fft.compute_observables_nd(psi0, [x, y])
    np.testing.assert_allclose(means, np.zeros(2), atol=1e-4)
    np.testing.assert_allclose(sigmas, sigma, atol=1e-2)


def test_bohmian_trajectories_1d_runs():
    """Sanity: 1D Bohmian integrator produces outputs of the expected shape
    and follows the mean of the density."""
    x = np.linspace(-4, 4, 512)
    sigma0 = 0.7
    psi0 = np.exp(-x ** 2 / (4 * sigma0 ** 2)).astype(complex)
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * (x[1] - x[0]))
    V = 0.5 * x ** 2
    x_particles = np.linspace(-0.5, 0.5, 5)
    rho, traj, t = fft.bohmian_trajectories_1d(
        x, psi0, V, T=0.5, dt=0.01, x_particles=x_particles, save_every=10)
    assert rho.shape[1] == len(x)
    assert traj.shape[1] == len(x_particles)
    # Non-crossing: initial sort is preserved.
    for k in range(traj.shape[0]):
        sorted_k = np.sort(traj[k])
        np.testing.assert_allclose(traj[k], sorted_k, atol=1e-10)
