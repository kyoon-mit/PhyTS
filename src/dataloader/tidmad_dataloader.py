"""
tidmad_dataset.py
=================
PyTorch Dataset for loading the processed TIDMAD HDF5 files produced by
process_tidmad.py.

Each processed file contains three aligned datasets:
    time_series_ch1  (N, L)  – raw SQUID time series chunks (noise + signal)
    time_series_ch2  (N, L)  – injected-signal-only chunks  (clean signal)
    signal_frequency (N, 1)  – identified signal frequency for each chunk

Design changes specific to this dataset
----------------------------------------
1. SHARED NORMALISATION: ch1 and ch2 are normalised by the same scale factor,
   derived from ch1 (the noisy channel).  This preserves the amplitude
   relationship between the noisy and clean signals — essential for denoising.

2. SUB-CHUNKING: an optional window_len / stride lets you slice each stored
   L-sample chunk into shorter overlapping or non-overlapping windows at load
   time.  This avoids re-running process_tidmad.py to change sequence length.

3. TRAIN/VAL SPLIT: the original TIDMAD files carry a "training" or
   "validation" label that is propagated into the processed files as a
   "source_split" attribute.  Pass split="train", "val", or "all".

4. FAST FREQUENCY LOOKUP: the freq → class-index map is built once at init
   using rounded keys, so __getitem__ does a fast O(1) dict lookup instead of
   a linear scan over all candidate frequencies.

Typical usage
-------------
    from tidmad_dataset import TIDMADDataset, worker_init_fn
    from torch.utils.data import DataLoader

    ds_train = TIDMADDataset(
        data_dir      = "./tidmad_output",
        split         = "train",
        window_len    = 65536,   # sub-chunk each 10M-sample chunk into windows
        stride        = 32768,   # 50 % overlap between consecutive windows
        normalize     = True,
        freq_as_index = True,
    )

    loader = DataLoader(
        ds_train,
        batch_size     = 32,
        shuffle        = True,
        num_workers    = 4,
        worker_init_fn = worker_init_fn,
    )

    for ch1, ch2, freq in loader:
        # ch1, ch2 : (batch, window_len)  float32 — same normalisation scale
        # freq     : (batch,)             int64 class index
        ...
"""

import os

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Main Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class TIDMADDataset(Dataset):
    """
    PyTorch Dataset for TIDMAD processed HDF5 files.

    Parameters
    ----------
    data_dir : str
        Directory containing tidmad_processed_*.h5 files.

    split : str
        Which subset of files to use.
        "train" – files derived from abra_training_*.h5 source files.
        "val"   – files derived from abra_validation_*.h5 source files.
        "all"   – use every processed file regardless of origin (default).
        Requires that process_tidmad.py wrote a "source_split" attribute to
        each output file.  Falls back to "all" if the attribute is absent.

    return_mode : str
        Which channel(s) to return.
        "ch1"  – raw SQUID channel only.
        "ch2"  – injected-signal channel only.
        "both" – both channels (default).

    window_len : int or None
        If given, each stored chunk of length L is sliced into windows of
        this many samples.  Must be <= L.  None returns the full L-sample
        chunk as a single item.

    stride : int or None
        Step between the start of consecutive windows.  Defaults to
        window_len (non-overlapping).  Set to a smaller value for overlap.
        Ignored when window_len is None.

    normalize : bool
        If True, apply shared normalisation: both ch1 and ch2 are divided by
        std(ch1_window) + eps.  This keeps the two channels in the same
        coordinate space, which is required for any denoising or regression
        task that uses both channels.  ch1 is additionally mean-subtracted.

    freq_as_index : bool
        If True, return the frequency label as a zero-based integer class
        index built from the sorted signal_freq_choices.  Useful for
        classification.  If False, return the raw Hz value as float32.

    """

    # Filename prefixes for each split — derived from the renamed output files:
    #   tidmad_training_0000.h5  …  tidmad_training_0019.h5
    #   tidmad_validation_0000.h5 … tidmad_validation_0019.h5
    _SPLIT_PREFIX = {
        "train": "tidmad_training_",
        "val":   "tidmad_validation_",
        "all":   "tidmad_",          # matches both prefixes
    }

    def __init__(
        self,
        data_dir:      str,
        split:         str        = "all",
        return_mode:   str        = "both",
        window_len:    int | None = None,
        stride:        int | None = None,
        normalize:     bool       = True,
        freq_as_index: bool       = False,
    ):
        if return_mode not in ("ch1", "ch2", "both"):
            raise ValueError(
                f"return_mode must be 'ch1', 'ch2', or 'both'; got '{return_mode}'")
        if split not in ("train", "val", "all"):
            raise ValueError(
                f"split must be 'train', 'val', or 'all'; got '{split}'")

        self.data_dir      = str(data_dir)
        self.split         = split
        self.return_mode   = return_mode
        self.normalize     = normalize
        self.freq_as_index = freq_as_index

        # Initialise early so __del__ / close() never sees a missing attribute,
        # even if __init__ raises before reaching the end of the method.
        self._file_handles: dict[str, h5py.File] = {}

        # ── Discover files for the requested split ────────────────────────
        # The split is encoded in the filename prefix, so no HDF5 attribute
        # lookup is needed.  "all" matches both "tidmad_training_" and
        # "tidmad_validation_" via the shared "tidmad_" prefix.
        prefix = self._SPLIT_PREFIX[split]
        self.paths = sorted([
            os.path.join(self.data_dir, fname)
            for fname in os.listdir(self.data_dir)
            if fname.startswith(prefix) and fname.endswith(".h5")
        ])
        if not self.paths:
            present = sorted(
                fn for fn in os.listdir(self.data_dir) if fn.endswith(".h5"))
            hint = (f"  .h5 files present: {present}"
                    if present else "  No .h5 files found in that directory.")
            raise FileNotFoundError(
                f"No files matching '{prefix}*.h5' found in '{data_dir}'.\n"
                f"{hint}\n"
                f"  Expected names like 'tidmad_training_0000.h5' or "
                f"'tidmad_validation_0000.h5'.")

        # ── Read shared metadata from the first file ──────────────────────
        with h5py.File(self.paths[0], "r") as f:
            self.chunk_length = int(f.attrs["chunk_length"])
            self.sample_rate  = float(f.attrs["sample_rate_hz"])
            # The sorted array of candidate frequencies passed to process_tidmad.py
            self.freq_choices = np.array(f.attrs["signal_freq_choices"],
                                         dtype=np.float32)

        # ── Resolve window / stride ───────────────────────────────────────
        # If window_len is not specified, treat each stored chunk as one item.
        if window_len is None:
            self.window_len = self.chunk_length
            self.stride     = self.chunk_length   # stride irrelevant for one window
        else:
            if window_len > self.chunk_length:
                raise ValueError(
                    f"window_len={window_len} exceeds chunk_length={self.chunk_length}")
            self.window_len = window_len
            self.stride     = stride if stride is not None else window_len

        # Number of windows that fit in one stored chunk
        self._windows_per_chunk = (
            (self.chunk_length - self.window_len) // self.stride + 1
        )

        # ── Build the flat index ──────────────────────────────────────────
        # Each entry is (file_path, local_chunk_row, window_offset).
        # window_offset is the sample index of the window start within the chunk.
        # When window_len == chunk_length, there is exactly one window per chunk
        # and window_offset is always 0.
        self._index: list[tuple[str, int, int]] = []

        for path in self.paths:
            with h5py.File(path, "r") as f:
                n_chunks = int(f["time_series_ch1"].shape[0])
            for chunk_row in range(n_chunks):
                for w in range(self._windows_per_chunk):
                    offset = w * self.stride
                    self._index.append((path, chunk_row, offset))

        # ── Build fast frequency → class-index lookup ─────────────────────
        # Frequencies are already snapped to signal_freq_choices by
        # process_tidmad.py, so we round to 6 decimal places to get exact
        # dictionary keys.  This makes __getitem__ an O(1) dict lookup rather
        # than a linear scan over all candidate frequencies.
        self._freq_to_idx: dict[float, int] = {
            round(float(f), 6): i
            for i, f in enumerate(sorted(self.freq_choices))
        }

        # (self._file_handles was initialised at the top of __init__
        #  before any code that could raise, so __del__ is always safe.)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_handle(self, path: str) -> h5py.File:
        """Return a cached open h5py.File for *path*, opening it if needed."""
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, "r")
        return self._file_handles[path]

    def _shared_normalize(
        self,
        ch1_win: np.ndarray,
        ch2_win: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalise ch1 and ch2 using shared statistics derived from ch1.

        Why shared?  ch1 = noise + signal, ch2 = signal only.  Dividing both
        by std(ch1) keeps them in the same amplitude coordinate space, so a
        model predicting ch2 from ch1 is learning a meaningful denoising map.
        Normalising independently would scale the two channels to the same
        apparent variance, erasing the SNR information.

        Transformation applied:
            ch1_out = (ch1 - mean(ch1)) / (std(ch1) + eps)
            ch2_out =  ch2             / (std(ch1) + eps)

        ch2 is not mean-subtracted because it is centred near zero (pure
        sinusoid); subtracting the ch1 mean from it would introduce a bias.
        """
        eps  = 1e-8
        mu1  = ch1_win.mean()
        std1 = ch1_win.std()
        ch1_norm = (ch1_win - mu1) / (std1 + eps)
        ch2_norm =  ch2_win       / (std1 + eps)
        return ch1_norm, ch2_norm

    def _lookup_freq_idx(self, freq_hz: float) -> int:
        """
        Convert a stored frequency value to its class index.

        Rounds to 6 decimal places to match the key format built at init,
        avoiding floating-point comparison errors in the hot path.
        """
        key = round(freq_hz, 6)
        if key in self._freq_to_idx:
            return self._freq_to_idx[key]
        # Fallback: nearest-neighbour search (should rarely be needed)
        nearest = min(self._freq_to_idx.keys(), key=lambda f: abs(f - freq_hz))
        return self._freq_to_idx[nearest]

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Total number of windows across all chunks and files."""
        return len(self._index)

    def __getitem__(self, idx: int):
        """
        Load and return one window.

        Returns
        -------
        Depending on return_mode and freq_as_index:
            (ch1, freq)       if return_mode == "ch1"
            (ch2, freq)       if return_mode == "ch2"
            (ch1, ch2, freq)  if return_mode == "both"

        ch1, ch2 : torch.FloatTensor  shape (window_len,)
        freq     : torch.FloatTensor  scalar Hz   (freq_as_index=False)
                   torch.LongTensor   scalar idx  (freq_as_index=True)
        """
        path, chunk_row, offset = self._index[idx]
        f = self._get_handle(path)

        # ── Read the frequency label for this chunk ────────────────────────
        # signal_frequency shape is (N, 1); index [chunk_row, 0] is a scalar.
        # All windows from the same chunk share the same frequency label.
        freq_hz = float(f["signal_frequency"][chunk_row, 0])

        # ── Slice the window from the stored chunk ────────────────────────
        # Read only the window slice rather than the full L-sample chunk to
        # keep per-item I/O proportional to window_len, not chunk_length.
        start = offset
        end   = offset + self.window_len

        if self.return_mode in ("ch1", "both"):
            ch1 = f["time_series_ch1"][chunk_row, start:end].astype(np.float32)

        if self.return_mode in ("ch2", "both"):
            ch2 = f["time_series_ch2"][chunk_row, start:end].astype(np.float32)

        # ── Normalisation ─────────────────────────────────────────────────
        # CHANGE from naive version: ch1 and ch2 use shared statistics so
        # that the amplitude relationship between them is preserved.
        if self.normalize:
            if self.return_mode == "both":
                ch1, ch2 = self._shared_normalize(ch1, ch2)
            elif self.return_mode == "ch1":
                # Normalise ch1 on its own when ch2 is not returned
                std1 = ch1.std()
                ch1  = (ch1 - ch1.mean()) / (std1 + 1e-8)
            elif self.return_mode == "ch2":
                # ch2 alone: normalise by its own std (no ch1 available)
                std2 = ch2.std()
                ch2  = ch2 / (std2 + 1e-8)

        # ── Encode the frequency label ────────────────────────────────────
        if self.freq_as_index:
            freq_tensor = torch.tensor(
                self._lookup_freq_idx(freq_hz), dtype=torch.long)
        else:
            freq_tensor = torch.tensor(freq_hz, dtype=torch.float32)

        # ── Return ────────────────────────────────────────────────────────
        if self.return_mode == "ch1":
            return torch.tensor(ch1, dtype=torch.float32), freq_tensor
        if self.return_mode == "ch2":
            return torch.tensor(ch2, dtype=torch.float32), freq_tensor
        return (
            torch.tensor(ch1, dtype=torch.float32),
            torch.tensor(ch2, dtype=torch.float32),
            freq_tensor,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dimension / metadata helpers
    # ─────────────────────────────────────────────────────────────────────────

    def input_len(self) -> int:
        """Length of each returned time-series tensor (window_len)."""
        return self.window_len

    def n_classes(self) -> int:
        """Number of distinct signal frequency classes."""
        return len(self._freq_to_idx)

    def freq_class_labels(self) -> list[float]:
        """Sorted list of Hz values corresponding to class indices 0, 1, …"""
        return sorted(self._freq_to_idx.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Resource management
    # ─────────────────────────────────────────────────────────────────────────

    def __getstate__(self):
        """
        Called by Python's pickle machinery when the dataset is serialised
        (e.g. when DataLoader copies it to worker processes).

        h5py.File objects cannot be pickled — they wrap a C-level file
        descriptor that has no meaning outside the process that opened it.
        We strip _file_handles from the state before pickling so that the
        copy sent to each worker contains no open handles.  The handles are
        re-opened lazily in each worker on the first __getitem__ call.
        """
        state = self.__dict__.copy()
        state["_file_handles"] = {}   # exclude open handles from the pickle
        return state

    def __setstate__(self, state):
        """
        Called by pickle when the dataset is deserialised in each worker.
        Restores all attributes and ensures _file_handles starts empty so
        that _get_handle() opens fresh connections in the new process.
        """
        self.__dict__.update(state)
        self._file_handles = {}       # always start with no open handles

    def close(self):
        """Explicitly close all open HDF5 file handles."""
        for fh in self._file_handles.values():
            fh.close()
        self._file_handles.clear()

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-worker DataLoader support
# ─────────────────────────────────────────────────────────────────────────────

def worker_init_fn(worker_id: int):
    """
    Pass as worker_init_fn= to DataLoader when num_workers > 0.

    HDF5 file handles are not fork-safe.  Each DataLoader worker inherits the
    parent's _file_handles dict, whose file descriptors are now shared between
    processes.  This function clears that dict in each worker so that handles
    are re-opened independently on first access.

    Usage:
        loader = DataLoader(dataset, num_workers=4,
                            worker_init_fn=worker_init_fn)
    """
    import torch.utils.data
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._file_handles = {}


# ─────────────────────────────────────────────────────────────────────────────
# Lightning DataModule
# ─────────────────────────────────────────────────────────────────────────────

class TIDMADDataModule(L.LightningDataModule):
    """Thin Lightning wrapper around TIDMADDataset.

    Maps Lightning stages onto the dataset's split argument:
      stage='fit'                → train=split('train'), val=split('val')
      stage in ('test','predict')→ test=split('val')

    Init args mirror TIDMADDataset; batch_size / num_workers / pin_memory are
    handled here.
    """

    def __init__(
        self,
        data_dir:      str,
        return_mode:   str        = "both",
        window_len:    int | None = None,
        stride:        int | None = None,
        normalize:     bool       = True,
        freq_as_index: bool       = False,
        batch_size:    int        = 32,
        num_workers:   int        = 0,
        pin_memory:    bool       = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self, split: str) -> TIDMADDataset:
        return TIDMADDataset(
            data_dir      = self.hparams.data_dir,
            split         = split,
            return_mode   = self.hparams.return_mode,
            window_len    = self.hparams.window_len,
            stride        = self.hparams.stride,
            normalize     = self.hparams.normalize,
            freq_as_index = self.hparams.freq_as_index,
        )

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train = self._make_dataset("train")
            self.val   = self._make_dataset("val")
        if stage in ("test", "predict") or stage is None:
            self.test  = self._make_dataset("val")

    def _loader(self, ds: TIDMADDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size     = self.hparams.batch_size,
            shuffle        = shuffle,
            num_workers    = self.hparams.num_workers,
            pin_memory     = self.hparams.pin_memory,
            worker_init_fn = worker_init_fn if self.hparams.num_workers > 0 else None,
        )

    def train_dataloader(self):   return self._loader(self.train, shuffle=True)
    def val_dataloader(self):     return self._loader(self.val,   shuffle=False)
    def test_dataloader(self):    return self._loader(self.test,  shuffle=False)
    def predict_dataloader(self): return self.test_dataloader()
