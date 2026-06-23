# EEG Headset Layouts for Motor Imagery BCIs in the Internet of Sounds

Companion repository for the paper **Benchmarking EEG Headset Layouts for Motor Imagery BCIs in the Internet of Sounds**.

This repository contains scripts and results for a topology-aware benchmark of wearable EEG headset layouts for left- vs. right-hand motor imagery (MI) decoding. Headset layouts are represented as virtual electrode masks over the public Cho2017 64-channel MI dataset and evaluated with reference pipelines in the MOABB framework.

The repository evaluates **electrode layouts**, not physical EEG products. It does not measure electrode technology, amplifier characteristics, fit stability, proprietary preprocessing, wireless robustness, latency, or long-term usability.

## Structure

```text
scripts/
  headset_profiles.py                 # headset collection and channel-mask utilities
  run_within_session.py               # performer-specific within-session benchmark
  run_cross_subject.py                # public-facing cross-subject LOSO benchmark
  check_channels.py                   # report channels missing from the Cho2017 montage
  make_heatmap.py                     # generate within/cross heatmaps from summary CSVs
  make_tradeoff_plot.py               # generate topology-performance trade-off plot
  make_electrode_occurrence_maps.py   # generate electrode-occurrence maps by form factor
  make_montage_legend.py              # generate the Cho2017 montage legend

results/
  CSV_performer_specific/                 # within-session CSV outputs
  CSV_public_facing/                      # cross-subject CSV outputs

figures/                              # generated paper figures
```

## Installation
```bash
pip install -r requirements.txt
```

MOABB will download/cache the Cho2017 dataset according to the local MOABB/MNE configuration.

## Running the benchmarks
The benchmark scripts run one headset mask at a time. To run a layout mask, change the `headset = HEADSETS[...]` line in the corresponding script.

```bash
python scripts/run_within_session.py
python scripts/run_cross_subject.py
```

The scripts save raw MOABB results and aggregated summary CSVs into:
```text
results/within_session/
results/cross_subject/
```

## Generating figures
```bash
python scripts/make_heatmap.py --csv results/within_session/__summary.csv --out figures/within_heatmap.png
python scripts/make_tradeoff_plot.py --within results/within_session/__summary.csv --cross results/cross_subject/__summary_CrossSub.csv --out figures/topology_tradeoff.png
python scripts/make_electrode_occurrence_maps.py
python scripts/make_montage_legend.py
```

## Notes
- ROC-AUC is the primary metric.
- Macro-F1 is computed as a secondary descriptive metric.
- Legacy 10-20 labels T3/T4/T5/T6 are normalized to T7/T8/P7/P8.
- Channels not available in the Cho2017 montage are omitted from the corresponding benchmark mask.
