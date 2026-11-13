"""Diagnose LinOSS (unpatched, damped IMEX) instability on LIGO v2.

Trains the v2 LinOSS model (H=128, P=128, 4 blocks) outside Lightning for a fixed number of
steps and logs, every --log_every steps: loss, pre-clip grad norm, and per block the SSM
parameter health (damping G, step size, fraction of oscillators at the A lower boundary,
largest transition-eigenvalue magnitude) and max |activation| after the block.
Stops at the first non-finite loss or gradient and reports which leaves / blocks blew up.

Variants (training-only, model unchanged):
  --ssm_lr_factor f   lr for A_diag, G_diag, steps = f * lr
  --ssm_no_wd         no weight decay on A_diag, G_diag, steps
  --warmup n          linear LR warmup over n steps

Usage: python diag_linoss.py --tag base --lr 1e-3 --steps 2000
"""

import argparse
import csv
import math
import sys
import time

sys.path.insert(0, "/home/barmstrong/TimeSeriesPhysics/src")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader

from dataloader.LIGO_dataloader import LIGODataset
from models.linoss import LinOSS
from tasks.LIGO.regression_jax import gaussian_nll_loss

SSM_NAMES = {"A_diag", "G_diag", "steps"}
D = "/fast/barmstrong/LIGO_2048Hz_v2"

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--ssm_lr_factor", type=float, default=1.0)
p.add_argument("--ssm_no_wd", action="store_true")
p.add_argument("--warmup", type=int, default=0)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--log_every", type=int, default=25)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="/fast/barmstrong/outputs/diag_linoss")
args = p.parse_args()
print(vars(args), flush=True)

torch.manual_seed(args.seed)
ds = LIGODataset(
    "train", f"{D}/train.h5", f"{D}/val.h5", f"{D}/test.h5",
    target_variables=("chirp_mass",), observed_variables=("snr",), injected_data_key="whitened_injected",
    strain_frequency=2048, strain_duration=4.5, window_begin=0.25, window_end=4.25, jitter=0.25,
)
dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=6, drop_last=True,
                generator=torch.Generator().manual_seed(args.seed), persistent_workers=True)

model = LinOSS(num_blocks=4, N=2, ssm_size=128, H=128, output_dim=2, task="regression",
               output_step=1, discretization="damped_IMEX", drop_rate=0.1, seed=args.seed)
state = eqx.nn.State(model)
params, static = eqx.partition(model, eqx.is_inexact_array)


def is_ssm(path):
    return any(getattr(k, "name", None) in SSM_NAMES for k in path)


labels = jax.tree_util.tree_map_with_path(lambda path, _: "ssm" if is_ssm(path) else "reg", params)


def sched(base):
    if args.warmup:
        return optax.join_schedules([optax.linear_schedule(0.0, base, args.warmup), optax.constant_schedule(base)], [args.warmup])
    return base


opt = optax.chain(
    optax.clip_by_global_norm(args.clip) if args.clip else optax.identity(),
    optax.multi_transform({
        "reg": optax.adamw(sched(args.lr), weight_decay=args.wd),
        "ssm": optax.adamw(sched(args.lr * args.ssm_lr_factor), weight_decay=0.0 if args.ssm_no_wd else args.wd),
    }, lambda _: labels),  # labels is an eqx Module (callable), so wrap it
)
opt_state = opt.init(params)


@eqx.filter_vmap(in_axes=(None, 0, None, 0), out_axes=(0, None), axis_name="batch")
def fwd(m, x, s, k):
    return m(x, s, key=k)


@eqx.filter_jit
def step(params, state, opt_state, x, y, key):
    def lossf(pr):
        out, ns = fwd(eqx.combine(pr, static), x, state, jax.random.split(key, x.shape[0]))
        return gaussian_nll_loss(out, y), ns
    (loss, ns), g = eqx.filter_value_and_grad(lossf, has_aux=True)(params)
    gnorm = optax.global_norm(g)
    upd, nos = opt.update(g, opt_state, params)
    return eqx.apply_updates(params, upd), ns, nos, loss, gnorm, g


def layer_stats(layer):
    """SSM health for one LinOSS layer (damped IMEX)."""
    dt = jax.nn.sigmoid(layer.steps)
    G = jax.nn.relu(layer.G_diag)
    lo = (2 + dt * G - 2 * jnp.sqrt(1 + dt * G)) / dt**2
    hi = (2 + dt * G + 2 * jnp.sqrt(1 + dt * G)) / dt**2
    A = lo + jax.nn.relu(layer.A_diag - lo) - jax.nn.relu(layer.A_diag - hi)
    S = 1 + dt * G
    m11, m12, m21, m22 = 1 / S, -dt / S * A, dt / S, 1 - dt**2 / S * A
    tr, det = m11 + m22, m11 * m22 - m12 * m21
    disc = (tr**2 - 4 * det).astype(jnp.complex64)
    lam = jnp.maximum(jnp.abs((tr + jnp.sqrt(disc)) / 2), jnp.abs((tr - jnp.sqrt(disc)) / 2))
    return dict(
        G_min=float(G.min()), G_zero=float(jnp.mean(G == 0)),
        dt_min=float(dt.min()), dt_max=float(dt.max()),
        A_at_lo=float(jnp.mean(layer.A_diag <= lo)),
        lam_max=float(lam.max()), n_undamped=int(jnp.sum(lam > 1 - 1e-4)),
    )


@eqx.filter_jit
def act_max(params, state, x):
    m = eqx.tree_inference(eqx.combine(params, static), value=True)

    def one(xi):
        h = jax.vmap(m.linear_encoder)(xi)
        outs = []
        for b in m.blocks:
            hn, _ = b.norm(h.T, state)
            z = b.ssm(hn.T)
            outs.append(jnp.max(jnp.abs(z)))
            h, _ = b(h, state, key=jax.random.key(0))
        return jnp.stack(outs)
    return jax.lax.map(one, x[:8])  # 8 samples is enough for magnitudes


f = open(f"{args.out}_{args.tag}.csv", "w", newline="")
w = None
key = jax.random.key(args.seed)
it, n, t0 = iter(dl), 0, time.time()
prev = (params, state)
while n < args.steps:
    try:
        X, y, _ = next(it)
    except StopIteration:
        it = iter(dl); continue
    x = jnp.asarray(X.numpy()).swapaxes(1, 2); yj = jnp.asarray(y.numpy())
    key, sk = jax.random.split(key)
    new_params, new_state, opt_state, loss, gnorm, g = step(params, state, opt_state, x, yj, sk)
    n += 1
    bad = not (math.isfinite(float(loss)) and math.isfinite(float(gnorm)))
    if n % args.log_every == 0 or bad or n == 1:
        row = dict(step=n, loss=float(loss), gnorm=float(gnorm), sec=round(time.time() - t0))
        amax = np.asarray(act_max(params, state, x)).max(axis=0)  # (samples, blocks) -> per block
        for i, blk in enumerate(eqx.combine(params, static).blocks):
            for k, v in layer_stats(blk.ssm).items():
                row[f"b{i}_{k}"] = v
            row[f"b{i}_act_max"] = float(amax[i])
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
        w.writerow(row); f.flush()
        short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                 if k in ("step", "loss", "gnorm", "sec") or k.endswith(("lam_max", "n_undamped", "act_max", "G_zero", "A_at_lo"))}
        print(short, flush=True)
    if bad:
        leaves = jax.tree_util.tree_leaves_with_path(g)
        badl = [jax.tree_util.keystr(pth) for pth, v in leaves if not bool(jnp.all(jnp.isfinite(v)))]
        print(f"NON-FINITE at step {n}: loss={float(loss)} gnorm={float(gnorm)}; non-finite grad leaves: {badl[:12]}", flush=True)
        break
    params, state = new_params, new_state
print(f"END tag={args.tag} steps={n} time={time.time() - t0:.0f}s", flush=True)
