# TESS data

Single-sector TESS light curves from the TESS-Gaia Light Curve pipeline, labelled with a stellar variability class and, for g-mode pulsators, a near-core rotation frequency (paper Section 3.2). Hosted in `TESS/` of [`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench).

## Download

```bash
python data/download.py --domain tess            # full dataset (~250 MB)
python data/download.py --domain tess --sample   # training splits only, copied to train, val and test
```

## Files

`data/TESS/split/tess_{classification,regression}_{train,val,test}.parquet`: fixed 80/10/10 splits, with no Gaia DR3 source in more than one split. The dataloader reads them directly.

## Format

One row per light curve:

| Column | Description |
|---|---|
| `GaiaID`, `TIC`, `sector` | Gaia DR3 and TESS Input Catalog identifiers, observing sector |
| `label` | variability class (classification files) |
| `frot`, `frot_err` | near-core rotation frequency and its error, 1/day (regression files) |
| `time`, `flux` | time stamps (BTJD) and normalized flux |

The 8 classes are APERIODIC, CONTACT_ROT, DSCT_BCEP, ECLIPSE, GDOR_SPB, INSTRUMENT/JUNK, RRLYR_CEPH and SOLARLIKE.
