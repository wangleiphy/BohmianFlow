"""Training loop for Fisher divergence score matching.

Optimises

    L_Fisher(theta) = E_{rho_theta}[|s_theta - grad ln rho_theta|^2]

by backpropagation through the discretised Bohmian trajectory (BPTT).  The
inner integrator and loss are defined in
:mod:`bohmian_score_matching.trajectory` and :mod:`bohmian_score_matching.core`.

Features retained in this release (exactly matching the PRL runs):

* Adam with optional global gradient clipping.
* Reduce-on-plateau learning-rate schedule.
* Multi-GPU data parallelism via ``jax.pmap`` when more than one device is
  visible.
* Stochastic subsampling of time checkpoints each epoch (``n_checkpoints``).
* Resume support through :mod:`bohmian_score_matching.checkpoint`.
* Every ``print_every`` epochs the script also evaluates diagnostics on a
  small independent batch: ``E_mean``, ``E_std``, ``min|det F|``, and the
  fraction of caustic-masked particles.  These appear in the log line and
  are parsed by :file:`scripts/extract_training_log.py` for Figure 3.
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


def _sample_initial(key, M, r0_mean, r0_cov, v0):
    X0 = jax.random.multivariate_normal(key, r0_mean, r0_cov, shape=(M,))
    V0 = jnp.tile(v0, (M, 1))
    return X0, V0


def train_fisher(score_fn, params, V_fn,
                 r0_mean, r0_cov, v0,
                 T, dt, n_epochs, M_train, lr,
                 *,
                 initial_score_fn=None,
                 n_checkpoints=None,
                 hbar=1.0, mass=1.0,
                 seed=42, print_every=50,
                 grad_clip=None,
                 caustic_threshold=0.1, target_clip=100.0,
                 grad_mode='jacfwd',
                 lr_patience=200, lr_factor=0.5, lr_min=1e-6,
                 resume_opt_state=None, resume_epoch=0, prev_losses=None,
                 checkpoint_every=0, checkpoint_path=None,
                 eval_batch_size=512,
                 args=None):
    """Minimise the Fisher divergence.

    Args:
        score_fn: (params, x, t) -> (d,) score network.
        params: dict containing the trainable key and any non-trainable
            entries needed by ``score_fn`` (e.g. ``'freqs'``).
        V_fn: classical potential, (d,) -> scalar.
        r0_mean, r0_cov, v0: moments of the initial Gaussian distribution.
        T, dt: trajectory length and leapfrog step.
        n_epochs, M_train, lr: optimiser knobs.
        initial_score_fn: s_0(x); defaults to the exact Gaussian score.
        n_checkpoints: number of timesteps per epoch that contribute to the
            loss (None = all n_steps+1 steps).  Each epoch draws a fresh
            uniform subset of non-zero steps (step 0 is always included).
        hbar, mass: physical constants.
        grad_clip: if set, clip the global gradient norm to this value.
        caustic_threshold: per-particle mask in the loss (|det F|).
        target_clip: cap on the absolute value of each score-target
            component.
        grad_mode: ``'jacfwd'`` or ``'jacrev'`` for grad_{x0} ln|det F|.
        lr_patience / lr_factor / lr_min: reduce-on-plateau schedule; set
            ``lr_patience=0`` to disable.
        resume_opt_state, resume_epoch, prev_losses: pass values from a
            loaded checkpoint to continue training.
        checkpoint_every, checkpoint_path, args: periodic checkpoint saving.
        eval_batch_size: particle count for the print-time diagnostics
            (E_mean, min|det F|).  Kept small to bound logging cost.

    Returns:
        params: updated parameters.
        losses: jnp.array of per-epoch losses from this call.
        diagnostics: dict with ``final_loss``, ``final_epoch``,
            ``opt_state``, ``loss_history``.
    """
    from bohmian_score_matching.core import fisher_loss, make_initial_score_fn
    from bohmian_score_matching.checkpoint import save_checkpoint

    if initial_score_fn is None:
        initial_score_fn = make_initial_score_fn(r0_mean, r0_cov)

    # Detect which params key is trainable.
    if 'film_params' in params:
        train_key = 'film_params'
    elif 'mlp_params' in params:
        train_key = 'mlp_params'
    else:
        raise ValueError(
            "params must contain 'film_params' or 'mlp_params'")

    baseline_params = {k: v for k, v in params.items() if k != train_key}

    # Optimiser: always chain(clip, adam) so opt_state layout is independent
    # of grad_clip setting.
    clip_val = grad_clip if grad_clip is not None else 1e10
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_val),
        optax.inject_hyperparams(optax.adam)(learning_rate=lr),
    )

    if resume_opt_state is not None:
        try:
            fresh = optimizer.init(params[train_key])
            if jax.tree.structure(fresh) != jax.tree.structure(resume_opt_state):
                raise ValueError("opt_state structure mismatch")
            opt_state = resume_opt_state
        except Exception:
            print("  WARNING: opt_state structure mismatch, re-initialising")
            opt_state = optimizer.init(params[train_key])
    else:
        opt_state = optimizer.init(params[train_key])

    use_plateau = lr_patience > 0
    current_lr = lr
    best_smoothed = float('inf')
    epochs_since_improve = 0

    n_devices = jax.device_count()
    use_pmap = n_devices > 1
    if use_pmap:
        if M_train % n_devices != 0:
            raise ValueError(
                f"M_train={M_train} must be divisible by n_devices={n_devices}")
        M_local = M_train // n_devices

    # --- loss-and-grad builders -------------------------------------------
    def _loss(train_params, X0, V0, ckpt_idx):
        p = {train_key: train_params, **baseline_params}
        return fisher_loss(
            p, X0, V0, score_fn, V_fn, T, dt, initial_score_fn,
            ckpt_idx=ckpt_idx, hbar=hbar, mass=mass,
            caustic_threshold=caustic_threshold, target_clip=target_clip,
            grad_mode=grad_mode,
        )

    @jax.jit
    def loss_and_grad(train_params, X0, V0, ckpt_idx):
        return jax.value_and_grad(_loss)(train_params, X0, V0, ckpt_idx)

    if use_pmap:
        @partial(jax.pmap, axis_name='batch', in_axes=(None, 0, 0, None))
        def loss_and_grad_pmap(train_params, X0_local, V0_local, ckpt_idx):
            loss_val, grads = jax.value_and_grad(_loss)(
                train_params, X0_local, V0_local, ckpt_idx)
            loss_val = jax.lax.pmean(loss_val, axis_name='batch')
            grads = jax.lax.pmean(grads, axis_name='batch')
            return loss_val, grads

        @jax.pmap
        def sample_pmap(key):
            return _sample_initial(key, M_local, r0_mean, r0_cov, v0)

    @jax.jit
    def opt_step(train_params, grads, opt_state):
        updates, new_state = optimizer.update(grads, opt_state, train_params)
        return optax.apply_updates(train_params, updates), new_state

    # --- print-time diagnostics: E_mean, min|det F|, masked fraction ------
    from bohmian_score_matching.trajectory import propagate_with_F
    from bohmian_score_matching.network import quantum_potential_batch
    M_eval = min(M_train, max(1, int(eval_batch_size)))

    @jax.jit
    def evaluate_at_print(train_params, key):
        p = {train_key: train_params, **baseline_params}
        X0e, V0e = _sample_initial(key, M_eval, r0_mean, r0_cov, v0)

        def single(x0, v0):
            return propagate_with_F(
                x0, v0, p, score_fn, V_fn, T, dt, hbar, mass)

        x_all, v_all, F_all, t_traj = jax.vmap(single)(X0e, V0e)
        det_F = jnp.linalg.det(F_all)
        min_abs_det = jnp.min(jnp.abs(det_F))
        particle_caustic = jnp.any(
            jnp.abs(det_F) < caustic_threshold, axis=1)
        masked_frac = jnp.mean(particle_caustic.astype(jnp.float64))

        def _energy_at_t(X, V, t_val):
            Q = quantum_potential_batch(score_fn, p, X, t_val, hbar, mass)
            KE = 0.5 * mass * jnp.sum(V ** 2, axis=1)
            PE = jax.vmap(V_fn)(X)
            return KE + PE + Q

        stride = max(1, int(T / dt) // 30)
        t_sub = t_traj[0, ::stride]
        x_sub = x_all[:, ::stride, :]
        v_sub = v_all[:, ::stride, :]
        E_all = jax.vmap(_energy_at_t, in_axes=(1, 1, 0), out_axes=1)(
            x_sub, v_sub, t_sub)
        E_mean_per_t = jnp.mean(E_all, axis=0)
        return {
            'E_mean': jnp.mean(E_mean_per_t),
            'E_std': jnp.std(E_mean_per_t),
            'min_abs_det_F': min_abs_det,
            'masked_frac': masked_frac,
        }

    # ---------------------------------------------------------------------
    d = r0_mean.shape[0]
    n_steps = int(T / dt)
    use_ckpt = n_checkpoints is not None and n_checkpoints < n_steps + 1

    print(f"Training: {n_epochs} epochs, M={M_train}, d={d}")
    if use_pmap:
        print(f"  devices={n_devices}, M_local={M_local} (pmap data-parallel)")
    print(f"  T={T:.4f}, dt={dt}, steps={n_steps}, "
          f"checkpoints={n_checkpoints if use_ckpt else n_steps+1}")
    print(f"  lr={lr}"
          f"{f', plateau(p={lr_patience}, f={lr_factor}, min={lr_min})' if use_plateau else ''}"
          f"{f', grad_clip={grad_clip}' if grad_clip else ''}")
    print(f"  caustic_threshold={caustic_threshold}, target_clip={target_clip}")

    all_losses = list(prev_losses) if prev_losses else []
    losses_this_run = []
    key = jax.random.PRNGKey(seed)
    start_epoch = resume_epoch
    total_epochs = start_epoch + n_epochs
    t0 = time.time()

    for epoch in range(start_epoch, total_epochs):
        key, subkey, ckpt_key = jax.random.split(key, 3)

        # Checkpoint-step subsampling for the loss.  Uniform draw of
        # non-zero steps, with step 0 always included so s_0 anchors the
        # target.
        if use_ckpt:
            u = jax.random.uniform(ckpt_key, shape=(n_checkpoints - 1,))
            rest = jnp.floor(u * n_steps).astype(jnp.int32) + 1
            ckpt_idx = jnp.sort(jnp.concatenate(
                [jnp.array([0], dtype=jnp.int32), rest]))
        else:
            ckpt_idx = None

        if use_pmap:
            device_keys = jax.random.split(subkey, n_devices)
            X0, V0 = sample_pmap(device_keys)
            loss_arr, grads_arr = loss_and_grad_pmap(
                params[train_key], X0, V0, ckpt_idx)
            loss = loss_arr[0]
            grads = jax.tree.map(lambda x: x[0], grads_arr)
        else:
            X0, V0 = _sample_initial(subkey, M_train, r0_mean, r0_cov, v0)
            loss, grads = loss_and_grad(params[train_key], X0, V0, ckpt_idx)

        params[train_key], opt_state = opt_step(
            params[train_key], grads, opt_state)

        losses_this_run.append(loss)
        all_losses.append(float(loss))

        # Reduce-on-plateau
        if use_plateau and len(all_losses) >= lr_patience:
            recent_mean = float(jnp.mean(jnp.array(all_losses[-lr_patience:])))
            if recent_mean < best_smoothed * 0.999:
                best_smoothed = recent_mean
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            if epochs_since_improve >= lr_patience:
                new_lr = max(current_lr * lr_factor, lr_min)
                if new_lr < current_lr:
                    print(f"  ReduceOnPlateau: lr {current_lr:.2e} -> "
                          f"{new_lr:.2e}")
                    current_lr = new_lr
                    opt_state[1].hyperparams['learning_rate'] = jnp.array(
                        current_lr, dtype=jnp.float64)
                best_smoothed = recent_mean
                epochs_since_improve = 0

        if (checkpoint_every > 0 and checkpoint_path
                and (epoch + 1) % checkpoint_every == 0):
            _p = {train_key: params[train_key], **baseline_params}
            base, ext = os.path.splitext(checkpoint_path)
            save_checkpoint(f'{base}_ep{epoch+1}{ext}',
                            _p, opt_state, all_losses, epoch + 1, args)

        if (epoch + 1) % print_every == 0 or epoch == start_epoch:
            elapsed = time.time() - t0
            from jax.flatten_util import ravel_pytree
            flat_g, _ = ravel_pytree(grads)
            g_norm = float(jnp.linalg.norm(flat_g))
            key, ek = jax.random.split(key)
            e = evaluate_at_print(params[train_key], ek)
            E_m = float(e['E_mean'])
            E_s = float(e['E_std'])
            m_det = float(e['min_abs_det_F'])
            mf = float(e['masked_frac'])
            print(f"Epoch {epoch+1}/{total_epochs}: loss={float(loss):.6e}, "
                  f"|grad|={g_norm:.4f}, "
                  f"{elapsed/(epoch-start_epoch+1):.2f}s/ep"
                  f"{f', lr={current_lr:.2e}' if use_plateau and current_lr != lr else ''}"
                  f" | E_mean={E_m:.4f}, E_std={E_s:.4f}"
                  f" | min|det F|={m_det:.4f}, masked={mf:.4%}")

    # Final checkpoint
    if checkpoint_every > 0 and checkpoint_path and n_epochs > 0:
        _p = {train_key: params[train_key], **baseline_params}
        save_checkpoint(checkpoint_path, _p, opt_state, all_losses,
                        total_epochs, args)

    total_elapsed = time.time() - t0
    print(f"\nTraining complete: {total_elapsed:.0f}s total, "
          f"{total_elapsed/max(n_epochs,1):.2f}s/epoch")

    params = {train_key: params[train_key], **baseline_params}
    diagnostics = {
        'final_loss': float(losses_this_run[-1]) if losses_this_run else float('nan'),
        'loss_history': jnp.array(losses_this_run) if losses_this_run else jnp.array([]),
        'opt_state': opt_state,
        'final_epoch': total_epochs,
    }
    return params, jnp.array(losses_this_run), diagnostics
