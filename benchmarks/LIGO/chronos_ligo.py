"""Chronos foundation-model benchmark on LIGO gravitational-wave data.

Modes
-----
zero_shot   — Frozen Chronos backbone; MLP head trained on embeddings.
finetune    — LoRA adapters in attention layers + MLP head trained jointly.

Both modes embed each IFO (H1, L1) independently and concatenate the
embeddings to form a 2*d_model feature vector before the regression head.

Usage
-----
  # activate the foundation-model env first:
  source benchmarks/foundation/.venv/bin/activate

  # zero-shot
  python benchmarks/LIGO/chronos_ligo.py --mode zero_shot \
      --model_size base --out_dir results/LIGO/chronos_zeroshot

  # fine-tune (LoRA, last 2 blocks)
  python benchmarks/LIGO/chronos_ligo.py --mode finetune \
      --finetune_strategy lora --finetune_n_blocks 2 \
      --out_dir results/LIGO/chronos_finetune

  # smoke test (2 batches)
  python benchmarks/LIGO/chronos_ligo.py --mode zero_shot --smoke_test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── path wiring: make the foundation sub-package importable ───────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO / "src"), str(_REPO / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from foundation.wrappers.chronos_wrapper import ChronosWrapper  # noqa: E402


# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("LIGO_DATA_DIR", "data/LIGO"))
SPLITS = {"train": "train/sig_combined_train.h5",
           "val":   "val/sig_combined_val.h5",
           "test":  "test/sig_combined_test.h5"}
INJECTED_KEY = "whitened_injected"   # (N, 2, 16384)
TARGET_KEY   = "chirp_mass"          # (N,)
STRAIN_FREQ  = 256                   # Hz
WIN_START    = 59                    # seconds
WIN_END      = 63                    # seconds
START_IDX    = WIN_START * STRAIN_FREQ   # 15104
END_IDX      = WIN_END   * STRAIN_FREQ   # 16128
L            = END_IDX - START_IDX       # 1024


# ── data loading ──────────────────────────────────────────────────────────────

def load_split(split: str, max_samples: int | None = None):
    """Return (X: np.float32 (N, 2, L), y: np.float32 (N,))."""
    path = DATA_DIR / SPLITS[split]
    with h5py.File(path, "r") as f:
        X = f[INJECTED_KEY][:max_samples, :, START_IDX:END_IDX].astype(np.float32)
        y = f[TARGET_KEY][:max_samples].astype(np.float32)
    return X, y


# ── embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(wrapper: ChronosWrapper, X: np.ndarray, batch_size: int = 64):
    """Embed both IFOs separately and concatenate → (N, 2*d_model)."""
    N = X.shape[0]
    all_embs = []
    for start in range(0, N, batch_size):
        chunk = X[start:start + batch_size]       # (B, 2, L)
        h1 = wrapper.embed(chunk[:, 0, :]).embeddings  # (B, d_model)
        l1 = wrapper.embed(chunk[:, 1, :]).embeddings  # (B, d_model)
        all_embs.append(np.concatenate([h1, l1], axis=-1))
    return np.concatenate(all_embs, axis=0)        # (N, 2*d_model)


# ── regression head ────────────────────────────────────────────────────────────

class EmbeddingMLP(nn.Module):
    def __init__(self, d_embed: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),     nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(
    train_emb: np.ndarray, train_y: np.ndarray,
    val_emb: np.ndarray,   val_y: np.ndarray,
    max_epochs: int = 200, batch_size: int = 256, lr: float = 1e-3,
    device: str = "cpu",
) -> EmbeddingMLP:
    d = train_emb.shape[-1]
    head = EmbeddingMLP(d).to(device)
    opt  = optim.AdamW(head.parameters(), lr=lr)
    crit = nn.MSELoss()

    trn_ds = TensorDataset(
        torch.from_numpy(train_emb).to(device),
        torch.from_numpy(train_y).to(device),
    )
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

    val_emb_t = torch.from_numpy(val_emb).to(device)
    val_y_t   = torch.from_numpy(val_y).to(device)

    best_val, best_state = float("inf"), None
    for epoch in range(max_epochs):
        head.train()
        for xb, yb in trn_dl:
            opt.zero_grad()
            crit(head(xb), yb).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val_loss = crit(head(val_emb_t), val_y_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{max_epochs}  val_mse={val_loss:.6f}")

    head.load_state_dict(best_state)
    return head


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(head: EmbeddingMLP, test_emb: np.ndarray, test_y: np.ndarray, device: str):
    head.eval()
    with torch.no_grad():
        pred = head(torch.from_numpy(test_emb).to(device)).cpu().numpy()
    rmse = float(np.sqrt(np.mean((pred - test_y) ** 2)))
    mae  = float(np.mean(np.abs(pred - test_y)))
    return {"rmse": rmse, "mae": mae}


# ── LoRA fine-tuning helper ────────────────────────────────────────────────────

def apply_lora_to_chronos(pipeline, rank: int = 4, n_blocks: int = 2):
    """Inject LoRA adapters into the last *n_blocks* attention projections."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        model = pipeline.model
        target_modules = []
        # Chronos-T5: encoder/decoder attention projections
        for name, _ in model.named_modules():
            if any(p in name for p in ("q_proj", "k_proj", "v_proj", "out_proj")):
                target_modules.append(name)
        # Use only last n_blocks worth of layers
        target_modules = target_modules[-n_blocks * 4:]
        cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=rank,
                         lora_alpha=32, lora_dropout=0.1,
                         target_modules=target_modules)
        pipeline.model = get_peft_model(model, cfg)
        print(f"LoRA injected into {len(target_modules)} modules (r={rank})")
    except ImportError:
        print("peft not installed; falling back to last-N block unfreezing.")
        _unfreeze_last_n(pipeline.model, n_blocks)


def _unfreeze_last_n(model, n: int):
    """Freeze all parameters; unfreeze last n transformer blocks."""
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = [m for m in model.modules()
              if hasattr(m, "self_attn") or "block" in type(m).__name__.lower()]
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfroze last {n} blocks — trainable params: {trainable:,}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "finetune"], default="zero_shot")
    parser.add_argument("--model_size", default="small",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--head_epochs",      type=int, default=200)
    parser.add_argument("--head_batch_size",  type=int, default=256)
    parser.add_argument("--head_lr",          type=float, default=1e-3)
    parser.add_argument("--finetune_strategy", choices=["frozen", "last_n", "lora"],
                        default="lora")
    parser.add_argument("--finetune_n_blocks", type=int, default=2)
    parser.add_argument("--lora_rank",         type=int, default=4)
    parser.add_argument("--out_dir",   default="results/LIGO/chronos")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Load only 2*embed_batch_size samples per split for quick testing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke = args.smoke_test
    max_n = args.embed_batch_size * 2 if smoke else None
    print(f"Loading data (smoke={smoke}) ...")
    X_tr, y_tr = load_split("train", max_n)
    X_va, y_va = load_split("val",   max_n)
    X_te, y_te = load_split("test",  max_n)
    print(f"  train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")

    print(f"Loading Chronos ({args.model_size}) ...")
    wrapper = ChronosWrapper()
    wrapper.load(device=args.device, model_size=args.model_size)

    if args.mode == "finetune":
        print(f"Fine-tune strategy: {args.finetune_strategy}")
        if args.finetune_strategy == "lora":
            apply_lora_to_chronos(wrapper.pipeline, args.lora_rank, args.finetune_n_blocks)
        elif args.finetune_strategy == "last_n":
            _unfreeze_last_n(wrapper.model, args.finetune_n_blocks)
        # else: "frozen" — same as zero-shot head training

    print("Extracting embeddings ...")
    emb_tr = extract_embeddings(wrapper, X_tr, args.embed_batch_size)
    emb_va = extract_embeddings(wrapper, X_va, args.embed_batch_size)
    emb_te = extract_embeddings(wrapper, X_te, args.embed_batch_size)
    print(f"  embedding dim: {emb_tr.shape[-1]}")

    print("Training regression head ...")
    head = train_head(
        emb_tr, y_tr, emb_va, y_va,
        max_epochs=args.head_epochs,
        batch_size=args.head_batch_size,
        lr=args.head_lr,
        device=args.device,
    )

    print("Evaluating on test set ...")
    metrics = evaluate(head, emb_te, y_te, args.device)
    print(f"  chirp_mass RMSE : {metrics['rmse']:.6f}")
    print(f"  chirp_mass MAE  : {metrics['mae']:.6f}")

    import json
    out = out_dir / "metrics.json"
    with open(out, "w") as fp:
        json.dump({"mode": args.mode, "model_size": args.model_size, **metrics}, fp, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
