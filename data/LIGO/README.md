# LIGO data

Simulated binary-neutron-star signals injected into real, whitened LIGO Hanford (H1) and Livingston (L1) background at 2048 Hz (paper Section 3.1). Hosted in `LIGO/` of [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench).

## Download

```bash
python data/download.py --domain ligo            # full dataset (~4 TB)
python data/download.py --domain ligo --sample   # one shard, copied to train, val and test
python data/LIGO/extract_window.py --src data/LIGO --dest data/LIGO
```

## Files

The Hub stores 64 s segments in shards under `data/LIGO/{train,val,test}/`. `extract_window.py` cuts a short window around the merger out of every shard and writes one file per split, `data/LIGO/{train,val,test}.h5`, which the dataloader reads.

## Format

Each event holds:

| Dataset | Description |
|---|---|
| `whitened_signal`, `whitened_bkg` | whitened injected waveform and detector background, H1 and L1 |
| `chirp_mass` | chirp mass [M☉], the regression target |
| `mass_1`, `mass_2`, `mass_ratio`, `redshift`, spins, sky position, `inclination`, `psi`, `phi_c` | source parameters (Table 4) |
| `snr`, `ifo_snrs` | network and per-detector SNR |

The extracted files store `whitened_injected`, the sum of signal and background, with the same per-event parameters.
