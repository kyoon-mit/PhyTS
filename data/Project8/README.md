# Project 8 data

Simulated cyclotron radiation emission spectroscopy (CRES) events from a resonant-cavity detector, generated with Kassiopeia and Locust (paper Section 3.4). Hosted in `Project8/` of [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench).

## Download

```bash
python data/download.py --domain project8            # full dataset (~44 GB)
python data/download.py --domain project8 --sample   # one file, copied to train, valid and test
```

## Files

`data/Project8/{train,valid,test}/*.hdf5`; the dataloader reads every file of a split folder.

## Format

One row per event, with I and Q traces sampled at 403 MHz:

| Dataset | Description |
|---|---|
| `output_ts_I`, `output_ts_Q` | in-phase and quadrature signal |
| `output_ts_{I,Q}_cav_noise` | frequency-dependent (cavity) noise, used in the paper |
| `output_ts_{I,Q}_gauss_noise` | white Gaussian noise |
| `energy_eV` | electron kinetic energy, the regression target |
| other fields | electron truth parameters: frequencies, pitch angle, position (Table 7) |

The model input is signal plus noise, for example `output_ts_I + output_ts_I_cav_noise`.
