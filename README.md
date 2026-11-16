# FDPG-AdCL: Adversarial Contrastive Learning with Dynamic Frequency-Patch Graphs for Point-Level Multivariate Time-Series Anomaly Detection

Linux runtime package for the current FDPG-AdCL implementation. The pipeline supports MSL, SMAP, SMD, SWaT and PSM, and runs representation training, global scoring, local reconstruction scoring, frequency-patch graph scoring, fusion and evaluation.

**Temporal protocol:** concatenate each entity's input train/test arrays in their supplied chronological order. For total length N, use [0, floor(N/2)) for the training region, [floor(N/2), floor(3N/5)) for validation, and [floor(3N/5), N) for testing. This implements the manuscript's chronological 5:1:4 ratio. Model fitting and preprocessing use the original normal-source portion within the training region; the normal-source cap is preserved. The validation and test intervals do not change with the seed. Test data form one continuous block. This package does not include datasets, trained weights or experimental results.

Checkpoints use a new split identifier. Old 60/10/30 checkpoints are rejected; train fresh checkpoints under this protocol. This code change does not reproduce or verify the manuscript's numerical results. Dataset-specific parameters are taken from manuscript Table A.1. The reported three-run numerical results have not been independently reproduced.

## Requirements

- Linux with Python 3.10 (original environment: Python 3.10.20).
- For GPU execution: an NVIDIA GPU and a driver compatible with the CUDA 11.8 PyTorch build.
- Python dependencies pinned in requirements.txt.

From the extracted FDPG-AdCL directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B run_pipeline.py --help
```

The pipeline defaults to CUDA. Use --device cpu for CPU execution; full training may be slow.

## Dataset layout

Pass the directory of the individual dataset to --data-root.

| Dataset | Required layout |
| --- | --- |
| MSL / SMAP | train/*.npy, test/*.npy, labeled_anomalies.csv |
| SMD | train/, test/, test_label/ with matching .txt or .csv entity files |
| PSM | train.csv, test.csv, test_label.csv |
| SWaT | swat_train.csv and swat_test.csv; supported alternative names are listed in src/data/loading.py |

SWaT input must be CSV, not an Excel workbook. Its test file must contain a supported attack-label column. Keep sensor order and chronological order consistent. Channel counts must match the dataset configuration. Obtain datasets from their providers under the applicable access terms.

## Run

Assuming PSM is stored at /data/PSM, preview the stages first:

```bash
python -B run_pipeline.py \
  --dataset psm --data-root /data/PSM \
  --output ./outputs/psm_seed42 --device cuda --seed 42 --dry-run
```

Dry-run checks configuration and entity discovery, then prints commands. It does not load all data, create run outputs or start training.

Run the full pipeline by omitting --dry-run:

```bash
python -B run_pipeline.py \
  --dataset psm --data-root /data/PSM \
  --output ./outputs/psm_seed42 --device cuda --seed 42
```

For a single SMD entity:

```bash
python -B run_pipeline.py \
  --dataset smd --data-root /data/SMD --entity machine-1-1.txt \
  --output ./outputs/smd_m11_seed42 --device cuda --seed 42
```

For MSL, SMAP and SMD, omit --entity to process all supported entity files in train/. Run each dataset separately using msl, smap, smd, swat or psm.

The output directory must not already exist. Before training, the runner checks data loading, channel counts and block lengths. Each stage writes a log; completed artifacts are retained on failure. The runner does not automatically resume partial runs. Use a new output directory for a fresh run.

--config selects a custom dataset YAML. The runner overrides data_root and seed explicitly. configs/branches.yaml contains only the RGCSF and FPLIR branch settings. No hyperparameter search or automatic multi-seed experiment is performed.

## Files and outputs

```text
FDPG-AdCL/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── configs/
│   ├── branches.yaml
│   └── five dataset YAML files
└── src/
    ├── data/
    ├── models/
    ├── losses/
    └── training, scoring, fusion and metric modules
```

src/aggregate_results.py provides score loading and entity aggregation. src/evaluate_threshold_channel_diagnostic.py implements the final fusion and evaluation used by the runner. These modules are runtime dependencies.

Each run produces checkpoints/, global/, local/, base/, graph/, final/, logs/ and run_state.json. The final dataset report is final/datasets/<dataset>_metrics.json. run_state.json records resolved configurations, commands, completed stages and errors.

## Validation scope

Core CPU tests and stage argument checks passed during preparation. The archive is intended for Linux, but a fresh Linux installation and complete GPU training have not been verified. The split has been aligned with the manuscript. Table A.1 parameters are checked against resolved configurations and stage arguments. Three-run numerical results have not been independently reproduced.

## Manuscript parameter mapping

The five dataset YAMLs contain the local score weights, encoder warmup settings and per-channel stem width from Table A.1. Their paper_pipeline section supplies base-fusion, graph and final boundary-fusion settings to run_pipeline.py. MSL uses an encoder FFT segment length of 16. Common branch settings do not override the dataset-specific local weights. Use the unified entry point to apply all of these values; individual stage CLIs retain their generic defaults unless explicitly overridden. Each invocation performs one seed; run three distinct seeds into separate output directories and average the resulting per-dataset metrics to obtain a three-run mean. The historical experiment seeds and numerical results are not reconstructed by this package.
