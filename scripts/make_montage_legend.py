import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from moabb.datasets import Cho2017

# 1) Load one raw run to retrieve electrode names.
dataset = Cho2017()
sub = dataset.subject_list[0]
data_sub = dataset.get_data(subjects=[sub])
sess_id = next(iter(data_sub[sub]))
run_id  = next(iter(data_sub[sub][sess_id]))
raw = data_sub[sub][sess_id][run_id].copy().pick("eeg")

# 2) Apply standard montage.
raw.drop_channels(["Iz", "P9", "P10"]) 
raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")

# 3) Prefixes mapped to extended scalp-region names.
prefix_full = {
    "Fp": "Fronto-polar",
    "AF": "Anterior frontal",
    "F":  "Frontal",
    "FT": "Fronto-temporal",
    "FC": "Fronto-central",
    "C":  "Central",
    "CP": "Centro-parietal",
    "TP": "Temporo-parietal",
    "T":  "Temporal",
    "P":  "Parietal",
    "PO": "Parieto-occipital",
    "O":  "Occipital",
}

# 4) Colors by prefix.
prefix_color = {
    "Fp": "#6A5ACD",
    "AF": "#7B68EE",
    "F":  "#4682B4",
    "FT": "#5F9EA0",
    "FC": "#2E8B57",
    "C":  "#228B22",
    "CP": "#B8860B",
    "TP": "#CD853F",
    "T":  "#A0522D",
    "P":  "#BC8F8F",
    "PO": "#8B0000",
    "O":  "#800000",
}

def get_prefix(ch):
    for p in sorted(prefix_full.keys(), key=len, reverse=True):
        if ch.startswith(p):
            return p
    return None

# --- STYLE PARAMETERS ---
FIGSIZE = (3.35, 4.2)    # una colonna, più alta per scalpo+legenda
COORD_SCALE = 0.9       # >1 spinge elettrodi verso il bordo
DOT_SIZE = 1             # small dots
LABEL_FS = 6             # electrode-label font size
LABEL_DY = 0.006         # vertical label offset
LEG_FS = 8               # legend font size
STROKE_W = 1.2
# legend layout
LEG_ROWS = 6             # rows
LEG_COLS = 2             # columns
LEG_XPAD = 0.0002          # left padding in legend-axis coordinates
LEG_YPAD_TOP = 0.90
# ---------------------------

# 5) Figure: scalp montage above, legend below.
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.02)
ax = fig.add_subplot(gs[0, 0])
ax_leg = fig.add_subplot(gs[1, 0])
ax_leg.axis("off")

# 6) Scalp plot without labels.
raw.info.plot_sensors(kind="topomap", show_names=False, sphere = 'eeglab', axes=ax, show=False, to_sphere=True)

# Retrieve sensor offsets from the MNE PathCollection.
sensor_coll = None
for coll in ax.collections:
    if hasattr(coll, "get_offsets"):
        off = coll.get_offsets()
        if off is not None and len(off) == len(raw.ch_names):
            sensor_coll = coll
            break
if sensor_coll is None:
    sensor_coll = next(c for c in ax.collections if hasattr(c, "get_offsets"))

offsets = sensor_coll.get_offsets()
sensor_coll.set_alpha(0.0)  # hide original MNE sensor dots

# radial scaling
offsets_scaled = offsets * COORD_SCALE

# 7) Electrode labels.
for ch_name, (x, y) in zip(raw.ch_names, offsets_scaled):
    p = get_prefix(ch_name)
    if p is None:
        continue
    t = ax.text(
        x, y + LABEL_DY, ch_name,
        ha="center", va="center",
        fontsize=LABEL_FS,
        color=prefix_color.get(p, "black"),
        zorder=4,
    )
    t.set_path_effects([pe.withStroke(linewidth=STROKE_W, foreground="white")])

# 8) Dots above labels.
ax.scatter(
    offsets_scaled[:, 0], offsets_scaled[:, 1],
    s=DOT_SIZE, c="black", linewidths=0, zorder=5
)

# 9) Bottom legend.
legend_items = ["Fp", "AF", "F", "FT", "FC", "C", "CP", "TP", "T", "P", "PO", "O"]

# grid in axis coordinates (0..1)
col_w = (1.0 - 2 * LEG_XPAD) / LEG_COLS
row_h = (LEG_YPAD_TOP - 0.08) / LEG_ROWS  # un minimo di margine sotto
dx_code = 0.10  # space between code and text (fraction of column width)

for i, p in enumerate(legend_items):
    r = i // LEG_COLS
    c = i % LEG_COLS

    x0 = LEG_XPAD + c * col_w
    y0 = LEG_YPAD_TOP - r * row_h

    ax_leg.text(
        x0, y0, p,
        color=prefix_color.get(p, "black"),
        fontsize=LEG_FS, fontweight="bold",
        transform=ax_leg.transAxes, va="top", ha="left"
    )
    ax_leg.text(
        x0 + dx_code, y0, prefix_full[p],
        color="black",
        fontsize=LEG_FS,
        transform=ax_leg.transAxes, va="top", ha="left"
    )

# 10) Vector export.
fig.savefig(Path(__file__).resolve().parents[1] / "figures" / "montage_legend.svg", bbox_inches="tight")
fig.savefig(Path(__file__).resolve().parents[1] / "figures" / "montage_legend.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: figures/montage_legend.svg and figures/montage_legend.pdf")
