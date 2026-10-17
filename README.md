# Distribution-Free Uncertainty Quantification for Structured Financial Prediction via Conformal Risk Control

## Overview

CMRC certifies a structured financial object — a volatility interval, a return interval and a
credit default set — under a single risk budget. One scalar is pushed through one strictly
increasing map per task, so the weighted composite risk is monotone by construction and the
`B/(n+1)` allowance of conformal risk control is spent once for all tasks. The budget split across
tasks is derived from equalised marginal width cost rather than tuned, and a regime-stratified
variant carries a declared one-parameter shift band across the calibration-to-test transition.

This repository holds the calibration layer, the point forecasters the protocol needs, the
evaluation grid, and a verification driver that executes each claim it can and records the ones it
cannot. `claim_to_code.json` maps every paper statement to the code that implements it,
`verification_report.json` records the executed status of each check, and `integrity_manifest.json`
digests the shipped files.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/coupled_monotone_risk/risks` | nested interval and set families, risk curves, isotonic cost curves, causal strata |
| `src/coupled_monotone_risk/coupling` | coupling maps, equalised-marginal allocation, composite crossing |
| `src/coupled_monotone_risk/calibration` | Algorithm 1, adaptive stratum weights, shift band, split/adaptive/quantile baselines |
| `src/coupled_monotone_risk/bounds` | Theorem 1 slack terms, budget-inflation bounds, alignment screen |
| `src/coupled_monotone_risk/data` | realized-library, equity and credit readers, block layout, regime taxonomy, causal strata |
| `src/coupled_monotone_risk/models` | HAR family, CHAR, realized GARCH, ARFIMA, PatchTST, selective state space, credit scorer |
| `src/coupled_monotone_risk/metrics` | QLIKE, MSE, MDA, MAD, Winkler, CRPS, Kupiec, Christoffersen, AUC/AP/KS/Brier/ECE |
| `src/coupled_monotone_risk/evaluation` | protocol grid, arm runner, tables, ablations, width trade-off, verification checks |
| `src/coupled_monotone_risk/training` | optimiser groups, cosine schedule with warmup, atomic checkpoints, supervised loop |
| `src/coupled_monotone_risk/cli` | `train`, `calibrate`, `evaluate`, `audit` |
| `configs` | protocol, per-layer task settings, backbone settings, study definitions |
| `assets/paper_claims.json` | transcription of the printed tables used by the manuscript-arithmetic checks |

## Installation

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

Conda:

```bash
conda env create -f environment.yml
conda activate cmrc
export PYTHONPATH=src
```

Docker:

```bash
docker build -t cmrc .
docker run --rm cmrc
```

## Data

Only the sources below were reachable from the build environment and matched their documentation.
Several layers of the paper depend on sources that could not be obtained; those layers are reported
as BLOCKED rather than substituted.

| Layer | Source | Local path under `CMRC_DATA_ROOT` | Size | SHA-256 |
| --- | --- | --- | --- | --- |
| credit | Statlog German Credit, UCI 144, CC BY 4.0 | `credit/german.data-numeric` | 102,000 B | `2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8` |
| credit | Default of Credit Card Clients, UCI 350, CC BY 4.0 | `credit/default of credit card clients.xls` | 5,539,328 B | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| volatility | Oxford-Man Realized Library, 21-index subset | `omrl/*.csv` | not distributed | — |
| returns | Amazon and Alphabet daily bars | `equity/amzn.csv`, `equity/googl.csv` | not distributed | — |

```bash
export CMRC_DATA_ROOT=/path/to/data
mkdir -p "$CMRC_DATA_ROOT"/credit
curl -L -o german.zip https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip
unzip -o german.zip -d "$CMRC_DATA_ROOT"/credit
curl -L -o taiwan.zip https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
unzip -o taiwan.zip -d "$CMRC_DATA_ROOT"/credit
python -m coupled_monotone_risk.cli.audit
```

The realized-library endpoint that served the archive no longer resolves, and no official
per-ticker endpoint for the two equity series could be verified from this environment. Both are
therefore excluded from `dataset_urls.txt` and their layers report BLOCKED. Lending Club returned
HTTP 403 and the FI-2010 page did not resolve to the benchmark, so neither is listed. Home Credit
requires a competition account.

## Protocol

Blocks are chronological: training, a 22-observation embargo, the calibration block, then the test
block. The volatility panel is scored inside the seven frozen regime windows of the paper; the
credit panels carry a single block, or the panel's own week buckets when available. Causal strata
for calibration are built from an expanding-window drawdown and a moving realized-variance
quantile; the regime taxonomy labels results and never enters calibration.

```bash
python -m coupled_monotone_risk.cli.calibrate \
  --config configs/protocol/cmrc.yaml --config configs/tasks/credit.yaml \
  --panels german taiwan --ablations
```

## Training

```bash
python -m coupled_monotone_risk.cli.train --config configs/study/main.yaml --panel taiwan \
  --checkpoint artefacts/scorer.pt
```

The paper reports no optimizer, learning rate or epoch count for the credit scorer. Those values are
marked `engineering_default` in `configs/study/main.yaml` and are not claimed as the paper's.

## Evaluation

```bash
python -m coupled_monotone_risk.cli.evaluate --config configs/study/main.yaml --layer all
python -m coupled_monotone_risk.cli.audit
pytest -q
```

No table cell is stored in the repository. Every reported number is produced by the run that prints it. Where a closed form exists,
the audit recomputes the reference value instead of calling the module under test.

## Verification status

`verification_report.json` separates `code_status` from `manuscript_status`. Unit, integration,
training, shape, gradient, regression and hygiene checks execute in this environment; the data
layers that need the closed archives report BLOCKED. Manuscript checks compare the printed tables
with each other and record the discrepancies that remain in the article.

The container build is reported NOT_RUN when no container runtime is present. Run the driver
twice: the integrity manifest digests the reports, so the first run writes the reports and the
second run records their final digests.

## Compute budget

The executed checks run on CPU with no GPU. The paper states no hardware, wall-clock or storage
figure, so none is invented here. Full-grid runs over the realized-library panel would need the
archive described above; the credit layer trains on a few thousand rows per epoch.

## Licence and provenance

Code is released under Apache-2.0. Datasets remain under their providers' terms, recorded in
`dataset_urls.txt`. `assets/paper_claims.json` transcribes values printed in the article and is used
only to audit the article's own arithmetic; no transcribed value is reported as a result of this
release.

## Limitations

The calibration guarantees are marginal, not conditional, and the regime evidence is descriptive.
The composite object is only as informative as the point forecasters underneath it, and the paper's
own utility-axis conclusion is parity rather than superiority. The width ratio in Section 4.7 is reported against the same reference Eq. (7) caps;
the audit recomputes that ratio from the allocation rather than trusting either number.
