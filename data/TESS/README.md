# TESS (PhyTS-bench) Data

Parquet files are downloaded from HuggingFace and placed here for the dataloader.

**Dataset:** [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench)

## Download

```bash
python data/download.py --domain tess        # full dataset (~194 MB)
python data/download.py --domain tess --sample   # sample file only
```

## Directory layout after download

```
data/TESS/
  tess_classification.parquet   # 25,935 light curves, 8-class labels
  tess_regression.parquet       # 4,183 light curves, rotation freq targets
```

## Dataset schema

### `tess_classification.parquet`

| Column   | Type                   | Description                        |
| -------- | ---------------------- | ---------------------------------- |
| `GaiaID` | int64                  | Gaia source id                     |
| `TIC`    | int64                  | TESS Input Catalog id              |
| `sector` | int64                  | TESS sector index                  |
| `label`  | string                 | Variability class (8 classes)      |
| `time`   | `large_list` of double | Time samples (aligned with `flux`) |
| `flux`   | `large_list` of double | Flux values (aligned with `time`)  |

### `tess_regression.parquet`

| Column     | Type                   | Description                                      |
| ---------- | ---------------------- | ------------------------------------------------ |
| `GaiaID`   | int64                  | Gaia source id                                   |
| `TIC`      | int64                  | TESS Input Catalog id                            |
| `sector`   | int64                  | TESS sector index                                |
| `frot`     | float64                | Rotation frequency target                        |
| `frot_err` | float64                | Uncertainty for `frot`                           |
| `time`     | `large_list` of double | Time samples (aligned with `flux`)               |
| `flux`     | `large_list` of double | Normalized flux (aligned with `time`)            |

Light-curve length varies per row (542–3,917 samples); the dataloader pads or
crops as needed. See `src/dataloader/tess_dataloader.py` for details.
