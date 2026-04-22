"""Train the score network on the d=4 Morse chain (PRL Figs. 3-5).

Reproduces the 45,000-epoch run whose final checkpoint backs Figures 3, 4
and 5 of the paper.  A single call with the defaults is equivalent to the
chain of 3 job IDs documented in CLAUDE.md (14253 -> 14568 -> 14932):
    4x GPU pmap, d=4, FiLM conditioning, n_freq=4, M=5000, lr=1e-3,
    T=pi, dt=0.01, caustic_threshold=0.1, target_clip=100,
    grad_clip=10.0, hidden_dims=[128, 128].

Example (single GPU, short smoke run):
    python scripts/train_morse.py --n-epochs 100 --M 128 --hidden-dims 32 32 \
        --checkpoint-path checkpoints/smoke.pkl

Resume from a saved checkpoint:
    python scripts/train_morse.py --resume checkpoints/smoke.pkl \
        --n-epochs 200
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Physics
    p.add_argument('--d', type=int, default=4)
    p.add_argument('--lam', type=float, default=0.3)
    p.add_argument('--T', type=float, default=float(jax.numpy.pi))
    p.add_argument('--dt', type=float, default=0.01)
    # Optimisation
    p.add_argument('--n-epochs', type=int, default=45000)
    p.add_argument('--M', type=int, default=5000, help='Particle batch size')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--n-checkpoints', type=int, default=None,
                   help='Time-step subsampling (None = all steps)')
    p.add_argument('--lr-patience', type=int, default=500)
    p.add_argument('--lr-factor', type=float, default=0.5)
    p.add_argument('--lr-min', type=float, default=1e-6)
    # Loss
    p.add_argument('--caustic-threshold', type=float, default=0.1)
    p.add_argument('--target-clip', type=float, default=100.0)
    p.add_argument('--grad-mode', choices=['jacfwd', 'jacrev'], default='jacfwd')
    # Network
    p.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128])
    p.add_argument('--n-freq', type=int, default=4)
    p.add_argument('--conditioning', choices=['concat', 'film'], default='film')
    # Plumbing
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--print-every', type=int, default=50)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--checkpoint-path', default='checkpoints/morse_d4.pkl')
    p.add_argument('--resume', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from bohmian_flow.potentials import morse_chain
    from bohmian_flow.baseline import make_baseline_score
    from bohmian_flow.network import (
        make_score_network, count_params,
    )
    from bohmian_flow.core import make_initial_score_fn
    from bohmian_flow.train import train_fisher
    from bohmian_flow.checkpoint import load_checkpoint

    system = morse_chain(d=args.d, lam=args.lam)
    d = system['r0_mean'].shape[0]
    print('=' * 70)
    print(system['label'])
    print(f'  T={args.T:.4f}, dt={args.dt}, '
          f'steps={int(args.T / args.dt)}, devices={jax.device_count()}')
    print('=' * 70)

    p0 = system['mass'] * system['v0']
    s_base = make_baseline_score(system['V_fn'], system['r0_mean'],
                                 system['r0_cov'], p0)

    init_fn, score_fn, _ = make_score_network(
        d, args.hidden_dims, args.n_freq, s_base,
        conditioning=args.conditioning,
    )

    resume_opt, resume_epoch, prev_losses = None, 0, None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        params = ckpt['params']
        resume_opt = ckpt['opt_state']
        resume_epoch = ckpt['epoch']
        prev_losses = ckpt['losses']
    else:
        params = init_fn(jax.random.PRNGKey(args.seed))
    print(f'  parameters: {count_params(params)}')

    s0 = make_initial_score_fn(system['r0_mean'], system['r0_cov'])

    params, losses, diag = train_fisher(
        score_fn, params, system['V_fn'],
        system['r0_mean'], system['r0_cov'], system['v0'],
        args.T, args.dt, args.n_epochs, args.M, args.lr,
        initial_score_fn=s0,
        n_checkpoints=args.n_checkpoints,
        hbar=system['hbar'], mass=system['mass'],
        seed=args.seed, print_every=args.print_every,
        grad_clip=args.grad_clip,
        caustic_threshold=args.caustic_threshold,
        target_clip=args.target_clip, grad_mode=args.grad_mode,
        lr_patience=args.lr_patience, lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        resume_opt_state=resume_opt, resume_epoch=resume_epoch,
        prev_losses=prev_losses,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=args.checkpoint_path,
        args=args,
    )
    print(f'Final loss: {diag["final_loss"]:.6e}')


if __name__ == '__main__':
    main()
