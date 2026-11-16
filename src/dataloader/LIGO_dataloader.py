import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from typing import Literal

Stage = Literal['train', 'val', 'test']


class LIGODataset(Dataset):
    def __init__(
        self,
        stage: Stage,
        train_file: str,
        val_file: str,
        test_file: str,
        target_variables: tuple[str, ...],
        observed_variables: tuple[str, ...] = (),
        injected_data_key: str = 'injected_data',
        clean_data_key: str | None = None,
        strain_frequency: int = 512,
        strain_duration: float = 64.0,
        window_begin: float = 0.0,
        window_end: float = 64.0,
        downsample_factor: int = 1,
        jitter: float = 0.0,
        jitter_seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.injected_data_key = injected_data_key
        self.clean_data_key = clean_data_key
        self.target_variables = target_variables
        self.observed_variables = observed_variables
        self.downsample_factor = downsample_factor
        self.dtype = dtype
        self._f = None

        self.file_path = {'train': train_file, 'val': val_file, 'test': test_file}[stage]

        start_f = window_begin * strain_frequency
        end_f = window_end * strain_frequency
        if not (float(start_f).is_integer() and float(end_f).is_integer()):
            raise ValueError(
                f'window_begin and window_end must produce integer sample indices. '
                f'Got start={start_f}, end={end_f}.'
            )
        self.start_idx = int(start_f)
        self.end_idx = int(end_f)

        # Timing jitter: shift the [window_begin, window_end) crop by up to +-jitter seconds.
        # Train draws a fresh shift per item; val/test use a fixed per-sample shift (seeded).
        jitter_f = jitter * strain_frequency
        if not float(jitter_f).is_integer():
            raise ValueError(f'jitter must produce an integer number of samples; got {jitter_f}.')
        self.max_shift = int(jitter_f)
        self.random_shift = stage == 'train'

        with h5py.File(self.file_path, 'r') as f:
            self.n_samples, self.n_ifos, self.full_len = f[injected_data_key].shape

        expected_len = int(strain_duration * strain_frequency)
        if self.full_len != expected_len:
            raise ValueError(
                f'Dataset length {self.full_len} does not match '
                f'strain_duration ({strain_duration}) * strain_frequency ({strain_frequency}) = {expected_len}.'
            )
        if self.start_idx - self.max_shift < 0 or self.end_idx + self.max_shift > self.full_len:
            raise ValueError(
                f'Window [{window_begin}, {window_end}) s +- jitter {jitter} s exceeds the stored '
                f'{strain_duration} s segment.'
            )
        rng = np.random.default_rng(jitter_seed + {'train': 0, 'val': 1, 'test': 2}[stage])
        self.fixed_shifts = rng.integers(-self.max_shift, self.max_shift + 1, size=self.n_samples)

    def _get_file(self):
        if self._f is None:
            self._f = h5py.File(self.file_path, 'r')
        return self._f

    def __len__(self):
        return self.n_samples

    def _shift(self, idx) -> int:
        if self.max_shift == 0:
            return 0
        if self.random_shift:
            return int(torch.randint(-self.max_shift, self.max_shift + 1, ()))
        return int(self.fixed_shifts[idx])

    def __getitem__(self, idx):
        f = self._get_file()
        shift = self._shift(idx)
        start, end = self.start_idx + shift, self.end_idx + shift
        seq = f[self.injected_data_key][idx, :, start:end:self.downsample_factor]
        X = torch.as_tensor(seq, dtype=self.dtype)
        if self.clean_data_key is not None:
            clean = f[self.clean_data_key][idx, :, start:end:self.downsample_factor]
            y = torch.as_tensor(clean, dtype=self.dtype)
        else:
            y = self._get_vars(f, self.target_variables, idx)
        z = self._get_vars(f, self.observed_variables, idx)
        return X, y, z

    def _get_vars(self, f, keys, idx) -> torch.Tensor:
        if keys:
            return torch.stack([torch.as_tensor(f[k][idx], dtype=self.dtype) for k in keys])
        return torch.empty(0, dtype=self.dtype)


class LIGODataModule(L.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        target_variables: tuple[str, ...],
        observed_variables: tuple[str, ...] = (),
        injected_data_key: str = 'injected_data',
        clean_data_key: str | None = None,
        strain_frequency: int = 512,
        strain_duration: float = 64.0,
        window_begin: float = 0.0,
        window_end: float = 64.0,
        downsample_factor: int = 1,
        jitter: float = 0.0,
        jitter_seed: int = 0,
        dtype: torch.dtype = torch.float32,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self._dataset_kwargs = dict(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            target_variables=target_variables,
            observed_variables=observed_variables,
            injected_data_key=injected_data_key,
            clean_data_key=clean_data_key,
            strain_frequency=strain_frequency,
            strain_duration=strain_duration,
            window_begin=window_begin,
            window_end=window_end,
            downsample_factor=downsample_factor,
            jitter=jitter,
            jitter_seed=jitter_seed,
            dtype=dtype,
        )
        self._loader_kwargs = dict(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle

    def setup(self, stage: str | None = None):
        if stage == 'fit':
            self.train_dataset = LIGODataset('train', **self._dataset_kwargs)
            self.val_dataset = LIGODataset('val', **self._dataset_kwargs)
        elif stage == 'validate':
            self.val_dataset = LIGODataset('val', **self._dataset_kwargs)
        elif stage in ('test', 'predict'):
            self.test_dataset = LIGODataset('test', **self._dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, **self._loader_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, **self._loader_kwargs)
