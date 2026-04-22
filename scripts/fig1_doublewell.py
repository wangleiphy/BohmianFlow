"""Figure 1: double-well schematic.

Two-step pipeline:
  1. Compute |psi(x, t)|^2 on a fine FFT grid + 100 exact Bohmian trajectories
     propagated with the exact Bohmian velocity v = (hbar/m) Im(psi'/psi).
     Saves all data to an NPZ.
  2. Plot the waterfall of density snapshots overlaid with the trajectories.

Example:
    python scripts/fig1_doublewell.py --data-out data/fig1_doublewell.npz \\
        --output figures/fig1_doublewell.png
    python scripts/fig1_doublewell.py --data data/fig1_doublewell.npz \\
        --output figures/fig1_doublewell.png   # replot only
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Physics
    p.add_argument('--a', type=float, default=2.0)
    p.add_argument('--D', type=float, default=1.0)
    p.add_argument('--sigma0', type=float, default=0.5)
    p.add_argument('--T', type=float, default=float(np.pi))
    p.add_argument('--dt', type=float, default=0.002)
    p.add_argument('--N-grid', type=int, default=4096)
    p.add_argument('--L', type=float, default=8.0)
    p.add_argument('--n-particles', type=int, default=100)
    p.add_argument('--save-every', type=int, default=10)
    # Plotting
    p.add_argument('--x-range', type=float, nargs=2, default=[-4.0, 4.0])
    p.add_argument('--density-scale', type=float, default=0.55)
    p.add_argument('--traj-alpha', type=float, default=0.45)
    p.add_argument('--traj-lw', type=float, default=0.5)
    p.add_argument('--density-alpha', type=float, default=0.55)
    p.add_argument('--figsize', type=float, nargs=2, default=[4.5, 4.2])
    p.add_argument('--dpi', type=int, default=300)
    # I/O
    p.add_argument('--data', default=None, help='Load pre-computed NPZ (replot)')
    p.add_argument('--data-out', default='data/fig1_doublewell.npz')
    p.add_argument('--output', default='figures/fig1_doublewell.png')
    return p.parse_args()


def compute(args):
    from bohmian_flow.fft import bohmian_trajectories_1d
    x = np.linspace(-args.L, args.L, args.N_grid)
    dx = x[1] - x[0]
    V = args.D * (x ** 2 / args.a ** 2 - 1.0) ** 2
    psi0 = np.exp(-x ** 2 / (4 * args.sigma0 ** 2)).astype(complex)
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)

    rng = np.random.default_rng(42)
    x_particles = np.sort(rng.normal(0.0, args.sigma0, size=args.n_particles))

    print(f'FFT+Bohmian: T={args.T:.4f}, dt={args.dt}, '
          f'steps={int(args.T / args.dt)}, N={args.N_grid}, '
          f'particles={args.n_particles}')
    rho_arr, traj_x, t_arr = bohmian_trajectories_1d(
        x, psi0, V, args.T, args.dt, x_particles, save_every=args.save_every)

    os.makedirs(os.path.dirname(args.data_out) or '.', exist_ok=True)
    np.savez(args.data_out,
             x_grid=x, t_arr=t_arr, rho=rho_arr, traj_x=traj_x, V_grid=V,
             a=args.a, D=args.D, sigma0=args.sigma0,
             T=args.T, dt=args.dt, n_particles=args.n_particles)
    print(f'Saved data to {args.data_out}')
    return args.data_out


def plot(npz_path, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    x_grid = d['x_grid']
    rho = d['rho']
    traj_x = d['traj_x']
    T = float(d['T'])
    a = float(d['a'])
    t_arr = d['t_arr']

    x_lo, x_hi = args.x_range
    mask = (x_grid >= x_lo) & (x_grid <= x_hi)
    x_plot = x_grid[mask]

    snap_times = np.array([0.0, T / 4.0, T / 2.0, 3.0 * T / 4.0, T])
    snap_indices = [int(np.argmin(np.abs(t_arr - t))) for t in snap_times]

    density_height = args.density_scale * (T / 4.0)
    rho_max = max(rho[si, mask].max() for si in snap_indices)

    fig, ax = plt.subplots(figsize=args.figsize)
    for si in snap_indices:
        base = t_arr[si]
        rho_scaled = rho[si, mask] / rho_max * density_height
        ax.fill_between(x_plot, base, base + rho_scaled,
                        color='#4A90D9', alpha=args.density_alpha,
                        edgecolor='none', zorder=3)
        ax.plot(x_plot, base + rho_scaled, color='#1a4a8a', lw=0.8, zorder=4)

    t_mask = t_arr <= T
    for j in range(traj_x.shape[1]):
        ax.plot(traj_x[t_mask, j], t_arr[t_mask], color='black',
                lw=args.traj_lw, alpha=args.traj_alpha, zorder=2)

    ax.axvline(a, color='gray', lw=0.4, alpha=0.3, zorder=0)
    ax.axvline(-a, color='gray', lw=0.4, alpha=0.3, zorder=0)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$t$', fontsize=12)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.04 * T, T + density_height + 0.02 * T)
    ax.tick_params(labelsize=10)
    ax.set_yticks(snap_times)
    ax.set_yticklabels([r'$0$', r'$T/4$', r'$T/2$', r'$3T/4$', r'$T$'])
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f'Saved to {args.output}')
    plt.close()


def main():
    args = parse_args()
    npz_path = args.data if args.data else compute(args)
    plot(npz_path, args)


if __name__ == '__main__':
    main()
