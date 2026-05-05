# TESS (PhyTS-bench) Data

This directory holds helpers for the TESS split of the **PhyTS-bench** dataset on
Hugging Face: a small schema peek script, a downloader, and a combined
visualization script. Parquet files are downloaded on demand and cached under
`.cache/` (typically not tracked by git).

## Directory layout

```
data/TESS/
  .cache/                    # Hub download cache (layout may vary)
    TESS/
      tess_regression.parquet
      tess_classification.parquet
  download_tess.py            # print schema + one example row per file
  visualize_tess.py         # histograms, light curves, time-coverage rasters
  README.md
```

## About the dataset

**PhyTS-bench** is a physics time-series benchmark; the **TESS** subset is built
from *TESS* light curves (variable targets), distributed as two Parquet files:
regression and classification.

- **Dataset (Hub):** [https://huggingface.co/datasets/PhyTS-team/PhyTS-bench](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench)
- **TESS files on Hub:** [TESS/](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench/tree/main/TESS)
  - `tess_regression.parquet` (~31 MB) — 4,183 rows
  - `tess_classification.parquet` (~194 MB) — 25,935 rows

Each row is one light-curve **segment**: identifiers (`GaiaID`, `TIC`, TESS
`sector`) plus aligned `time` and `flux` as **variable-length** `double` lists
(Arrow `large_list<double>`). Lengths differ between rows, so training code must
pad, pack, or crop as needed.

### Light-curve lengths (samples per row)

For the cached PhyTS-bench export, `len(time)` == `len(flux)` per row with:


| File                          | Min length | Max length | # Lightcurves |
| ----------------------------- | ---------- | ---------- | ------------- |
| `tess_regression.parquet`     | 683        | 3,917      | 4,183         |
| `tess_classification.parquet` | 542        | 3,917      | 25,935        |

If the Hub Parquet changes, re-measure lengths with a short `pyarrow` scan (or update this table manually).

## Step 1 — Install optional Hub / Parquet / plot deps

Scripts need **pyarrow**, **huggingface_hub**, and **matplotlib**, listed under
the **jax** optional extra in `pyproject.toml`:

```bash
uv sync --extra jax
# or: pip install -e ".[jax]"
```

Set **HF_TOKEN** in the environment if you want better Hugging Face rate
limits (optional for public data).

## Step 2 — Schema peek

```bash
python data/TESS/download_tess.py
```

Downloads each file into `data/TESS/.cache/` if needed, then prints the Arrow
schema, row counts, and one sample row per file.

## Step 3 — Visualizations (`visualize_tess.py`)

One script writes:

- `**tess_histograms.png**` — classification `label` / `sector`, regression
`sector` / `frot`
- `**tess_regression_lightcurves.png**`, `**tess_classification_lightcurves.png**`
— random example segments; the figure subtitle reports the **median sampling
step Δt** (same units as the time column, plus an approximate minutes value if
time is days)
- `**tess_*_time_coverage.png`** (+ `**_preview.png**`) — per-row raster of
cadence occupancy on a fixed Δt grid (default 1500 bins), scalar sector index +
Matplotlib **colorbar** (tick labels `Sector …`, same idea as `nan_counting.py`),
x-axis **Cadence #** (0 … n−1 via `extent`); the PNG uses a **downsampled** grid
for Matplotlib (caps in the script) to keep RAM low—see the figure subtitle.

By default, PNGs are written to **`data/TESS/figures`**. With **`--where engaging`**
(or any data layout where Parquets live under `…/TESS/.cache/TESS/`), the default
output is **`…/data_engaging/TESS/figures`** on the pool (sibling of `.cache`).
Use **`--out-dir`** to override.

```bash
python data/TESS/visualize_tess.py
python data/TESS/visualize_tess.py --where engaging
# optional: python data/TESS/visualize_tess.py --out-dir path/to/figures
# optional: python data/TESS/visualize_tess.py --cache-dir path/to/cache  # uses path/to/cache/TESS
```

## Parquet column reference

### `tess_regression.parquet`


| Column     | Type                   | Description                                      |
| ---------- | ---------------------- | ------------------------------------------------ |
| `GaiaID`   | int64                  | Gaia source id                                   |
| `TIC`      | int64                  | TESS Input Catalog id                            |
| `sector`   | int64                  | TESS sector index                                |
| `frot`     | float64                | Rotation frequency target (regression)           |
| `frot_err` | float64                | Uncertainty for `frot`                           |
| `time`     | `large_list` of double | Time samples (aligned with `flux`)               |
| `flux`     | `large_list` of double | Normalized / relative flux (aligned with `time`) |


### `tess_classification.parquet`


| Column   | Type                   | Description                        |
| -------- | ---------------------- | ---------------------------------- |
| `GaiaID` | int64                  | Gaia source id                     |
| `TIC`    | int64                  | TESS Input Catalog id              |
| `sector` | int64                  | TESS sector index                  |
| `label`  | string                 | Variability / object class label   |
| `time`   | `large_list` of double | Time samples (aligned with `flux`) |
| `flux`   | `large_list` of double | Flux values (aligned with `time`)  |


## Notes

- **Variable length:** `time` and `flux` are the same length within a row, but
that length may change from row to row.
- **On-disk size:** the classification file is much larger than the regression
file; first run with a good network (or a warm Hub cache) may take a few
minutes.
- **Training / dataloading:** if you add a PyTorch Lightning (or other) pipeline
for TESS, document it here and point to the module under `src/` the same
way `data/TIDMAD/README.md` references `TIDMADDataModule`.

