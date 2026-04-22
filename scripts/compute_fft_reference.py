"""Compute the FFT split-operator reference for the Morse chain.

Saves an NPZ with per-mode means/sigmas and the exact total energy that
Figures 3, 4, and 5 compare against.

Example:
    python scripts/compute_fft_reference.py -o data/fft_ref_morse_d4.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d', type=int, default=4)
    p.add_argument('--N', type=int, default=64,
                   help='grid points per dimension')
    p.add_argument('--L', type=float, default=8.0,
                   help='grid half-range [-L, L]')
    p.add_argument('--T', type=float, default=float(np.pi))
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--D', type=float, default=12.5)
    p.add_argument('--beta', type=float, default=0.2)
    p.add_argument('--lam', type=float, default=0.3)
    p.add_argument('--q0', type=float, default=1.0)
    p.add_argument('-o', '--output', default='data/fft_ref_morse_d4.npz')
    p.add_argument('--save-psi-at', type=float, nargs='*', default=None,
                   help='optional times at which to cache psi (requires a '
                        'separate .pkl).')
    p.add_argument('--psi-pkl', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from bohmian_flow.fft import (
        gaussian_wavepacket_nd, split_operator_nd,
    )

    d = args.d
    grids = [np.linspace(-args.L, args.L, args.N) for _ in range(d)]

    omega0 = args.beta * np.sqrt(2.0 * args.D)
    sigma0_mode = np.sqrt(1.0 / (2.0 * omega0))
    r0 = np.zeros(d)
    r0[0] = args.q0
    sigma = np.full(d, sigma0_mode)

    psi0 = gaussian_wavepacket_nd(grids, r0, sigma)
    mem_mb = psi0.nbytes / 1e6
    print(f'FFT grid: d={d}, N={args.N}, {mem_mb:.0f} MB / wavefunction')

    def V_fn(*mesh):
        V = sum(args.D * (1.0 - np.exp(-args.beta * mesh[i])) ** 2
                for i in range(d))
        V = V + args.lam * sum(mesh[i] * mesh[i + 1] for i in range(d - 1))
        return V

    result = split_operator_nd(
        V_fn, grids, psi0, args.T, args.dt,
        checkpoint_every=1,
        save_psi_at=args.save_psi_at,
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(args.output,
             t=result['t'],
             means=result['means'],
             sigmas=result['sigmas'],
             E_exact=result['E_exact'],
             d=d, N=args.N, L=args.L, T=args.T, dt=args.dt,
             D=args.D, beta=args.beta, lam=args.lam, q0=args.q0)
    print(f'Saved moments + E_exact={result["E_exact"]:.6f} to {args.output}')

    if args.psi_pkl and result['psi_snapshots']:
        import pickle
        os.makedirs(os.path.dirname(args.psi_pkl) or '.', exist_ok=True)
        with open(args.psi_pkl, 'wb') as f:
            pickle.dump({
                'psi_snapshots': result['psi_snapshots'],
                'grids': grids,
                'k_grids': result['k_grids'],
                'd': d, 'T': args.T, 'dt': args.dt,
            }, f)
        print(f'Saved psi snapshots to {args.psi_pkl}')


if __name__ == '__main__':
    main()
