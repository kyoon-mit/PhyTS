"""Save a checkpoint for a model that requires no training (e.g. classical filters).

Instantiates the model from a YAML config and saves its state_dict in the
same checkpoint format used by Lightning (so load_model() in eval_pipeline.py
can load it transparently).

Usage:
    python tools/save_checkpoint.py \
        --cfg  configs/toy/train_toy_classical_denoising.yaml \
        --out  checkpoints/toy_classical_denoising/best.ckpt
"""

import argparse
import importlib
import yaml
import torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True, help='YAML config path')
parser.add_argument('--out', required=True, help='Output checkpoint path')
args = parser.parse_args()

with open(args.cfg) as f:
    cfg = yaml.safe_load(f)

model_cfg   = cfg['model']['init_args']['model']
class_path  = model_cfg['class_path']
init_args   = model_cfg.get('init_args', {})

module_name, class_name = class_path.rsplit('.', 1)
cls   = getattr(importlib.import_module(module_name), class_name)
model = cls(**init_args)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({'state_dict': {f'model.{k}': v for k, v in model.state_dict().items()}}, out)
print(f'Saved checkpoint → {out}  ({sum(p.numel() for p in model.parameters())} params)')
