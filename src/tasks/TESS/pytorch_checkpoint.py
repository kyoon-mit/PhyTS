"""TESS training helpers: PyTorch ``ModelCheckpoint`` with cluster-friendly dirpath.

LightningCLI (jsonargparse) does not support CLI flags such as
``--trainer.callbacks[1].init_args.dirpath``. Slurm scripts can set
``TESS_PYTORCH_CKPT_DIR`` to override the ``dirpath`` in YAML instead.
"""

import os
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint


class TESSPyTorchModelCheckpoint(ModelCheckpoint):
    """Same as :class:`ModelCheckpoint`; ``TESS_PYTORCH_CKPT_DIR`` overrides ``dirpath``.

    Parameters
    ----------
    dirpath
        Default directory from YAML. Ignored if ``TESS_PYTORCH_CKPT_DIR`` is set.
    **kwargs
        Forwarded to :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def __init__(self, dirpath: str | Path | None = None, **kwargs: Any):
        # TESS_PYTORCH_CKPT_DIR: HPC / Slurm; see module docstring.
        override = os.environ.get("TESS_PYTORCH_CKPT_DIR")
        if override is not None and override != "":
            dirpath = override
        super().__init__(dirpath=dirpath, **kwargs)
