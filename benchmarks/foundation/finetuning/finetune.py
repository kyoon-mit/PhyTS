"""
Fine-tuning adapters for foundation models.

Scope
-----
Fine-tuning is intrinsically model-specific: MOMENT exposes structured
task heads, Chronos needs LoRA on its T5 decoder, MOIRAI / Lag-Llama use
GluonTS predictors with their own trainers, and TimeMoE/TimesFM have
custom transformer blocks.  A fully-general, per-model fine-tuning
framework would require deep, brittle hooks into each library.

This module intentionally covers a narrower but useful slice:

1. ``finetune_embedding_regression``
    The zero-shot embedding+regression path already trains an MLP head
    on frozen embeddings.  Fine-tuned mode has three adapter strategies:

      * ``"frozen"``        — head only (same as zero-shot reference).
      * ``"last_n"``        — unfreeze the last N transformer blocks
                              (default N=2) and train jointly.
      * ``"lora"``          — inject LoRA adapters into attention
                              projection linear layers and train only
                              those + the head.  The original backbone
                              weights stay frozen.

    Falls back to frozen-backbone training if the wrapper doesn't expose
    its backbone in a standard way.

The forecasting / denoising fine-tuning paths are out of scope of this
first cut — they require per-model task-head wiring that isn't
compatible with the unified wrapper interface.  Use
``wrapper.pipeline`` / ``wrapper.model`` directly if you need to go
deeper for a single model.

Entry point
-----------
``finetune`` is the main dispatcher, imported lazily by
``run_benchmark.py`` so that a missing torch/lightning at zero-shot time
doesn't break anything.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# ────────────────────────────────────────────────────────────────────────────
# LoRA — low-rank adapters for nn.Linear layers
# ────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for an nn.Linear that adds a LoRA path.

    The base weight/bias are frozen; only ``lora_A`` and ``lora_B`` are
    trainable.  Forward computes ``base(x) + (x · Aᵀ) · Bᵀ · (α/r)`` so
    that initialisation (B = 0) yields zero adaptation, meaning the
    pretrained model's behaviour is preserved until training begins.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert r > 0, "LoRA rank must be positive"
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # A: (r, in_features), B: (out_features, r)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays at zero → BA = 0 at init → no behaviour change

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze the base layer
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        # (..., in) @ Aᵀ → (..., r); then @ Bᵀ → (..., out)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return result + delta * self.scaling


# Default attention-projection name patterns.  Covers T5 (Chronos),
# Llama-style (TimeMoE), BERT-style, and the generic "qkv" fusion used
# by TimesFM / MOMENT / MOIRAI backbones.
_LORA_DEFAULT_PATTERNS: Tuple[str, ...] = (
    r"\.(q|k|v|o)_proj$",          # Llama / LLaMA-like
    r"\.(q|k|v|o)$",               # T5 (SelfAttention.q, .k, .v, .o)
    r"\.qkv_proj$",                # fused QKV
    r"\.query$", r"\.key$", r"\.value$",  # BERT-style
    r"\.out_proj$",                # nn.MultiheadAttention
    r"\.dense$",                    # HF BERT attention output
    r"\.qkv$",                     # ViT-style
)


def _compile_patterns(patterns: Sequence[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def apply_lora(
    module: nn.Module,
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_patterns: Sequence[str] = _LORA_DEFAULT_PATTERNS,
    last_n_blocks: Optional[int] = None,
) -> List[nn.Parameter]:
    """Wrap matching nn.Linear layers with LoRALinear; return LoRA params.

    * ``target_patterns`` are regexes matched against the full dotted name
      of each submodule (e.g. ``layers.23.self_attn.q_proj``).  Defaults
      cover the common attention-projection names across HF and custom
      architectures.
    * ``last_n_blocks`` optionally restricts application to the last N
      top-level transformer blocks (children named ``layers.*`` /
      ``blocks.*`` / ``h.*`` / ``encoder.layer.*``) so earlier layers
      stay untouched — useful for small fine-tuning budgets.
    * Freezes everything first, then adds LoRA params (which are the
      only trainables).
    """
    # 1. Freeze everything
    for p in module.parameters():
        p.requires_grad_(False)

    # 2. Figure out which blocks are in-scope if last_n_blocks is set
    block_prefixes: Optional[List[str]] = None
    if last_n_blocks is not None:
        block_regex = re.compile(r"^(layers|blocks|h|encoder\.layer)\.(\d+)$")
        block_names: List[Tuple[int, str]] = []
        for name, _ in module.named_modules():
            m = block_regex.match(name)
            if m:
                block_names.append((int(m.group(2)), name))
        if block_names:
            block_names.sort()
            keep = block_names[-last_n_blocks:]
            block_prefixes = [n + "." for _, n in keep]

    compiled = _compile_patterns(target_patterns)

    def _in_scope(qualified: str) -> bool:
        if block_prefixes is None:
            return True
        return any(qualified.startswith(p) for p in block_prefixes)

    # 3. Walk named_modules and collect Linear targets
    targets: List[Tuple[str, nn.Linear]] = []
    for qualified, sub in module.named_modules():
        if not isinstance(sub, nn.Linear):
            continue
        if not _in_scope(qualified):
            continue
        if any(rx.search(qualified) for rx in compiled):
            targets.append((qualified, sub))

    # 4. Replace them in-place via parent attribute assignment
    for qualified, sub in targets:
        parent_name, _, child_name = qualified.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, LoRALinear(sub, r=r, alpha=alpha, dropout=dropout))

    # 5. Return the LoRA params that are now trainable
    lora_params: List[nn.Parameter] = []
    for sub in module.modules():
        if isinstance(sub, LoRALinear):
            lora_params.append(sub.lora_A)
            lora_params.append(sub.lora_B)
    return lora_params


# ────────────────────────────────────────────────────────────────────────────
# Backbone discovery — wrappers store the underlying nn.Module in slightly
# different attributes; we probe a handful of common locations.
# ────────────────────────────────────────────────────────────────────────────

_BACKBONE_PATHS = (
    ("model",),                      # TimesFM, TimeMoE, MOIRAI
    ("pipeline", "model"),           # Chronos (ChronosPipeline has .model)
    ("pipelines", "forecasting"),    # MOMENT (dict of pipelines per task)
    ("_module",),                    # MOIRAI (uni2ts MoiraiModule)
    ("_predictor",),                 # Lag-Llama estimator
)


def _find_backbone(wrapper) -> Optional[nn.Module]:
    for path in _BACKBONE_PATHS:
        obj = wrapper
        ok = True
        for attr in path:
            if isinstance(obj, dict):
                obj = obj.get(attr)
            else:
                obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj
    return None


def _unfreeze_last_n(module: nn.Module, n: int) -> List[nn.Parameter]:
    """Unfreeze the last `n` nn.Module children and return their params.

    We walk the top-level children and take the last `n` that actually
    contain parameters.  This is a heuristic that works well for the
    transformer-stacked architectures used by all six foundation models.
    """
    # Freeze everything first
    for p in module.parameters():
        p.requires_grad_(False)

    children = [c for c in module.modules() if isinstance(c, (nn.TransformerEncoderLayer,
                                                              nn.TransformerDecoderLayer))]
    if not children:
        # Fallback: just take the last few direct children that have params
        children = [c for c in module.children() if any(p.numel() for p in c.parameters())]

    targets = children[-n:] if n < len(children) else children
    trainable: List[nn.Parameter] = []
    for t in targets:
        for p in t.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


# ────────────────────────────────────────────────────────────────────────────
# Embedding + MLP regression head, fine-tuned jointly
# ────────────────────────────────────────────────────────────────────────────

class _RegressionHead(nn.Module):
    """Same architecture as the zero-shot MLP head so results are comparable."""

    def __init__(self, d_embed: int, n_targets: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def finetune_embedding_regression(
    wrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    target_idx: List[int],
    device: str = "cuda",
    epochs: int = 20,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    adapter: str = "last_n",                # "frozen" | "last_n" | "lora"
    unfreeze_last_n: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_last_n_blocks: Optional[int] = None,  # None = apply across whole backbone
    max_train_batches: Optional[int] = None,
    verbose: bool = True,
) -> nn.Module:
    """Fine-tune backbone (via ``adapter``) + MLP head for parameter regression.

    ``adapter`` selects the adaptation strategy:

    * ``"frozen"`` — no backbone params trainable (head-only baseline).
    * ``"last_n"`` — unfreeze the last ``unfreeze_last_n`` transformer
      blocks.
    * ``"lora"``   — inject LoRA adapters (``lora_r``, ``lora_alpha``,
      ``lora_dropout``) into attention projection Linear layers and
      train only those.  Apply over the full backbone or, if
      ``lora_last_n_blocks`` is set, only the final N blocks.

    Returns the trained MLP head.  The wrapper's backbone is modified in
    place: parameters are unfrozen, or Linear layers are wrapped.  Call
    ``wrapper.unload()`` afterwards to reset.
    """
    if not wrapper.supports_embed:
        raise ValueError(f"{wrapper.name} does not support embed(); cannot fine-tune.")

    backbone = _find_backbone(wrapper)
    trainable_backbone_params: List[nn.Parameter] = []

    if backbone is None:
        if verbose:
            print(f"  [finetune] Could not locate backbone for {wrapper.name}; "
                  "frozen-backbone fallback (head only).")
    elif adapter == "frozen":
        for p in backbone.parameters():
            p.requires_grad_(False)
        if verbose:
            print(f"  [finetune] {wrapper.name}: backbone fully frozen (head only).")
    elif adapter == "last_n":
        try:
            trainable_backbone_params = _unfreeze_last_n(backbone, unfreeze_last_n)
            if verbose:
                n = sum(p.numel() for p in trainable_backbone_params)
                print(f"  [finetune] {wrapper.name}: unfroze last "
                      f"{unfreeze_last_n} blocks — {n/1e6:.2f}M params trainable.")
        except Exception as e:
            if verbose:
                print(f"  [finetune] _unfreeze_last_n failed for {wrapper.name}: {e}")
    elif adapter == "lora":
        try:
            trainable_backbone_params = apply_lora(
                backbone, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                last_n_blocks=lora_last_n_blocks,
            )
            n = sum(p.numel() for p in trainable_backbone_params)
            n_modules = sum(1 for m in backbone.modules() if isinstance(m, LoRALinear))
            if verbose:
                scope = (f"last {lora_last_n_blocks} blocks"
                         if lora_last_n_blocks else "whole backbone")
                print(f"  [finetune] {wrapper.name}: LoRA r={lora_r}, "
                      f"α={lora_alpha} injected into {n_modules} attention "
                      f"projections ({scope}) — {n/1e6:.3f}M LoRA params trainable.")
                if n_modules == 0:
                    print(f"  [finetune] WARNING: no attention projections matched "
                          "the default patterns; head-only training will run.")
        except Exception as e:
            if verbose:
                print(f"  [finetune] apply_lora failed for {wrapper.name}: {e}")
    else:
        raise ValueError(
            f"Unknown adapter '{adapter}'. Choose from: frozen, last_n, lora."
        )

    # Determine embedding dim by running one batch through .embed()
    with torch.no_grad():
        first = next(iter(train_loader))
        x0 = first[0] if isinstance(first, (list, tuple)) else first["sig_bkg"]
        emb0 = wrapper.embed(x0[:1].numpy().astype(np.float32)).embeddings
        d_embed = emb0.shape[-1]

    head = _RegressionHead(d_embed, n_targets=len(target_idx)).to(device)
    optim = torch.optim.AdamW([
        {"params": trainable_backbone_params, "lr": backbone_lr},
        {"params": head.parameters(),         "lr": head_lr},
    ], weight_decay=1e-4)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    for ep in range(epochs):
        head.train()
        losses: List[float] = []
        for i, batch in enumerate(train_loader):
            if max_train_batches is not None and i >= max_train_batches:
                break
            sig_bkg = batch[0] if isinstance(batch, (list, tuple)) else batch["sig_bkg"]
            params_  = batch[2] if isinstance(batch, (list, tuple)) else batch["params"]
            y = params_[:, target_idx].float().to(device)

            # Re-embed through the (partially unfrozen) backbone
            emb = wrapper.embed(sig_bkg.numpy().astype(np.float32)).embeddings
            emb_t = torch.from_numpy(emb).to(device)
            pred = head(emb_t)
            loss = torch.nn.functional.mse_loss(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))

        # Validation
        head.eval()
        with torch.no_grad():
            val_losses: List[float] = []
            for batch in val_loader:
                sig_bkg = batch[0] if isinstance(batch, (list, tuple)) else batch["sig_bkg"]
                params_ = batch[2] if isinstance(batch, (list, tuple)) else batch["params"]
                y = params_[:, target_idx].float().to(device)
                emb = wrapper.embed(sig_bkg.numpy().astype(np.float32)).embeddings
                pred = head(torch.from_numpy(emb).to(device))
                val_losses.append(float(torch.nn.functional.mse_loss(pred, y).item()))
        vl = float(np.mean(val_losses)) if val_losses else float("inf")
        if verbose:
            tl = float(np.mean(losses)) if losses else 0.0
            print(f"  [finetune] epoch {ep+1}/{epochs}  train {tl:.4f}  val {vl:.4f}")
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return head


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────────

def finetune(
    wrapper,
    *,
    train_loader,
    val_loader,
    tasks: Iterable[str],
    target_idx: Optional[List[int]] = None,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str = "cuda",
    context_len: int = 512,
    horizon: int = 128,
    adapter: str = "last_n",
    unfreeze_last_n: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_last_n_blocks: Optional[int] = None,
    **_unused,
):
    """Dispatch fine-tuning per task.

    Currently only ``embedding`` is implemented.  ``adapter`` chooses the
    backbone-adaptation strategy: ``"frozen"``, ``"last_n"`` (default),
    or ``"lora"``.  For LoRA, the ``lora_*`` kwargs control rank, scale,
    dropout, and whether to restrict to the last N blocks.  Forecasting
    / denoising fine-tuning paths print a "not implemented" message so
    the caller can fall through to zero-shot evaluation.
    """
    tasks = set(tasks)
    results = {}

    if "embedding" in tasks and wrapper.supports_embed:
        if target_idx is None:
            raise ValueError("target_idx required for embedding fine-tuning")
        head = finetune_embedding_regression(
            wrapper, train_loader, val_loader,
            target_idx=target_idx, device=device, epochs=epochs,
            backbone_lr=lr * 1e-1,  # 10× smaller than head LR
            head_lr=lr,
            adapter=adapter,
            unfreeze_last_n=unfreeze_last_n,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_last_n_blocks=lora_last_n_blocks,
        )
        results["embedding_head"] = head

    if "forecasting" in tasks:
        print(f"  [finetune] forecasting fine-tuning not implemented for "
              f"{wrapper.name}; zero-shot will run.")
    if "denoising" in tasks:
        print(f"  [finetune] denoising fine-tuning not implemented for "
              f"{wrapper.name}; zero-shot will run.")

    return results
