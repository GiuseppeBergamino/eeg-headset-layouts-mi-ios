"""
Single-panel trade-off plot (within vs cross-subject) for virtual EEG headsets.

Inputs:
  - within summary CSV (e.g., __summary.csv)
  - cross-subject summary CSV (e.g., __summary_CrossSub.csv)

Output:
  - single figure: x = N_ch, y = ROC-AUC (%)
    • One marker per headset at within best-of-five AUC
    • One arrow per headset from within best-of-five to cross best-of-five
    • Color = Form factor (Cap/Helmet/Headband/Headphones)
    • Shape = circle if price known, square if price unavailable
    • Size = qualitative cost tier (4 levels)
    • FULL_CHO plotted as a star (reference)

Price/cost input:
  Prefer adding these optional columns to the WITHIN CSV (per headset row):
    - PriceKnown  (0/1 or True/False)   -> marker shape (circle vs square)
    - CostTier    (1..4)               -> marker size
  If absent, the script falls back to the dictionaries PRICE_KNOWN / COST_TIER below.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import textwrap
from matplotlib.lines import Line2D


# -----------------------------
# User-editable fallbacks
# -----------------------------

FORM_ORDER = ["Cap", "Helmet", "Headband", "Headphones"]

# If your CSVs do NOT contain PriceKnown/CostTier columns,
# you can fill these fallbacks (keys are Mask codes).
# Any Mask missing here will default to PriceKnown=True and CostTier=2.
PRICE_KNOWN: Dict[str, bool] = {
    # "BITBRAIN_AIR": True,
    # "SOME_DEVICE": False,
}

COST_TIER: Dict[str, int] = {
    # 1=Low, 2=Mid, 3=High, 4=Very high
    # "BITBRAIN_AIR": 2,
    # "SOME_DEVICE": 4,
}

# Marker sizes (scatter uses area in points^2).
COST_TIER_SIZES = {1: 160, 2: 300, 3: 450, 4: 600}
COST_TIER_LABELS = {1: "Low", 2: "Mid", 3: "High", 4: "Very high"}

# Identify reference montage mask in your CSVs.
REFERENCE_MASK = ""

# Mapping from internal headset codes to display names
HEADSET_NAME_MAP = {
    "FULL_CHO": "Reference Cho2017",
    "BITBRAIN_DIADEM": "Bitbrain Diadem",
    "BITBRAIN_HERO": "Bitbrain Hero",
    "BITBRAIN_AIR": "Bitbrain Air",
    "BITBRAIN_IKON": "Bitbrain Ikon",
    "BRAINBIT_DRAGON": "BrainBit Dragon",
    "BRAINBIT_HEADBAND_PRO": "BrainBit\nHeadbandPro",
    "BRAINBIT_HEADBAND": "BrainBit Headband",
    "BRAINBIT_HEADPHONES": "BrainBit Headphones",
    "CGX_QUICK_32R": "CGX-Quick-32R",
    "CGX_QUICK_20R": "CGX-Quick-20R",
    "EMOTIV_FLEX": "Emotiv\nFlex 2",
    "EMOTIV_EPOCH_X": "Emotiv\nEPOC X",
    "EMOTIV_EPOC_X": "Emotiv\nEPOC X",
    "EMOTIV_INSIGHT": "Emotiv Insight",
    "EMOTIV_MN8": "Emotiv\nMN8",
    "GTEC_UNICORN_HYBRID": "g.tec\nUnicorn-Hybrid",
    "MUSE_2_HEADBAND": "Muse 2 Headband",
    "MUSE_S_ATHENA": "Muse S Athena",
    "MINDROVE_VISION": "Mindrove Vision",
    "MINDROVE_ARC": "Mindrove Arc",
    "MINDROVE_LUCID": "Mindrove Lucid",
    "MINDROVE_BRIGHT": "Mindrove Bright",
    "NEEURO_SENZEBAND": "Neeuro SenzeBand",
    "NEUROSITY_CROWN": "Neurosity Crown",
    "NEUROSKY_MINDWAVE_2": "Neurosky Mindwave 2",
    "ULTRACORTEX_16": "Ultracortex(16)",
    "ULTRACORTEX_8": "Ultracortex(8)",
    "WS_DSI_24": "W. S.\nDSI-24",
    "WS_DSI_7": "W. S.\nDSI-7",
    "WS_DSI_VR300": "W.S.\nDSI VR300",
    "WS_DSI_VRVEP": "W.S.\nDSI VRVEP",
    "ABM_B_ALERT_X24": "A.B.M. B-Alert X24",
    "ABM_B_ALERT_X10": "A.B.M. B-Alert X10",
}

def wrap_label(name: str, width: int = 12) -> str:
    """Soft-wrap headset names for in-marker labels.
    If the name already contains explicit newlines, preserve them and wrap each
    line separately. This allows manual '\n' breaks in HEADSET_NAME_MAP.
    """
    s = str(name)
    if "\n" in s:
        parts = s.splitlines()
        wrapped = [
            textwrap.fill(p, width=width, break_long_words=False, break_on_hyphens=False)
            for p in parts
        ]
        return "\n".join(wrapped)
    return textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=False)


def pretty_mask_name(mask: str) -> str:
    return HEADSET_NAME_MAP.get(mask, mask.replace("_", " ").title())

# -----------------------------
# Parsing helpers
# -----------------------------

PM_RE = re.compile(
    r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:±|\+/-|\+-)\s*([0-9]+(?:[.,][0-9]+)?)\s*$"
)


def parse_mean_std(cell) -> Tuple[float, float]:
    """Parse 'mean±std' strings; returns (mean, std). NaN-safe."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return (np.nan, np.nan)
    s = str(cell).strip()
    m = PM_RE.match(s)
    if not m:
        # allow plain numbers (no ±) as mean
        try:
            v = float(s.replace(",", "."))
            return (v, np.nan)
        except Exception as e:
            raise ValueError(f"Cannot parse mean±std from: {cell!r}") from e
    mean_s = m.group(1).replace(",", ".")
    std_s = m.group(2).replace(",", ".")
    return float(mean_s), float(std_s)


def infer_auc_scale(values: np.ndarray) -> float:
    """
    If values look like fractions (0..1.2), return 100.0 else 1.0.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 1.0
    med = float(np.median(v))
    return 100.0 if med <= 1.2 else 1.0


def auc_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_AUC")]
    if not cols:
        raise ValueError("No AUC columns found (expected columns ending with '_AUC').")
    return cols


def best_of_five(df: pd.DataFrame, auc_cols: List[str], scale_mode: str = "auto") -> pd.Series:
    """
    Compute per-row best ROC-AUC across pipelines (means only).
    Returns a Series with best mean AUC (scaled to percent if needed).
    """
    means = []
    for c in auc_cols:
        m = df[c].apply(lambda x: parse_mean_std(x)[0])
        means.append(m.to_numpy(dtype=float))
    mat = np.vstack(means).T  # shape (n_rows, n_pipes)

    # scale
    if scale_mode == "auto":
        factor = infer_auc_scale(mat.flatten())
    elif scale_mode == "percent":
        factor = 1.0
    elif scale_mode == "fraction":
        factor = 100.0
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    mat = mat * factor

    return pd.Series(np.nanmax(mat, axis=1), index=df.index)


def deterministic_jitter(x: np.ndarray, max_jitter: float = 0.22) -> np.ndarray:
    """
    Deterministically spread points that share the same x across [-max_jitter, +max_jitter].
    """
    x = np.asarray(x, dtype=float)
    out = x.copy()
    # group indices by integer channel count
    uniq = sorted(set(out.tolist()))
    for xv in uniq:
        idx = np.where(out == xv)[0]
        n = idx.size
        if n <= 1:
            continue
        offsets = np.linspace(-max_jitter, max_jitter, n)
        out[idx] = out[idx] + offsets
    return out


# -----------------------------
# Main plotting
# -----------------------------

def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Mask", "Form", "N_ch"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}; found {list(df.columns)}")
    df = df.copy()
    df["N_ch"] = pd.to_numeric(df["N_ch"], errors="coerce")
    df["Form"] = pd.Categorical(df["Form"], categories=FORM_ORDER, ordered=True)
    # keep first occurrence per mask
    df = df.drop_duplicates(subset=["Mask"], keep="first").reset_index(drop=True)
    return df


def get_price_fields(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Return (price_known, cost_tier) per row.

    Priority:
      1) CSV columns: PriceKnown (bool-ish), CostTier (1..4)
      2) CSV column: Price (numeric) -> derive PriceKnown + 4 qualitative tiers
      3) Fallback dicts: PRICE_KNOWN / COST_TIER
    """
    # 1) Explicit columns (if you ever add them later)
    if "PriceKnown" in df.columns:
        pk = df["PriceKnown"].astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])
    else:
        pk = None

    if "CostTier" in df.columns:
        ct = pd.to_numeric(df["CostTier"], errors="coerce")
    else:
        ct = None

    if pk is not None and ct is not None:
        ct = ct.fillna(2).astype(int).clip(1, 4)
        return pk.astype(bool), ct

    # 2) Your current CSVs have "Price"
    if "Price" in df.columns:
        price = pd.to_numeric(df["Price"], errors="coerce")
        pk = price.notna()

        # default tier for missing prices (shape will be square anyway)
        ct = pd.Series(2, index=df.index, dtype=int)

        p = price[pk].astype(float)
        # Use log-scale binning to reduce the influence of extreme outliers and obtain
        # more stable qualitative tiers (we never expose numeric prices in the paper).
        p_for_bins = np.log2(p.clip(lower=1.0))

        if len(p_for_bins) >= 4 and p_for_bins.nunique() >= 2:
            # Quantile-based binning into up to 4 groups, without exposing price values
            # duplicates='drop' avoids errors if many equal prices
            q = pd.qcut(p_for_bins, q=4, labels=False, duplicates="drop")  # 0..k-1
            k = int(q.max()) + 1

            if k == 1:
                ct.loc[pk] = 2
            else:
                # Rescale 0..k-1 -> 1..4
                ct.loc[pk] = np.round(q * (3.0 / (k - 1)) + 1).astype(int)
        else:
            # If too few prices, keep everything as mid tier
            ct.loc[pk] = 2

        return pk.astype(bool), ct.clip(1, 4)

    # 3) Fallback dicts
    pk = df["Mask"].map(lambda m: PRICE_KNOWN.get(m, True)).astype(bool)
    ct = df["Mask"].map(lambda m: COST_TIER.get(m, 2)).fillna(2).astype(int).clip(1, 4)
    return pk, ct



def build_form_colors(df: pd.DataFrame, cmap_name: str = "tab10") -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap(cmap_name)
    forms_present = [f for f in FORM_ORDER if f in df["Form"].astype(str).unique().tolist()]
    colors = {}
    for i, f in enumerate(forms_present):
        colors[f] = cmap(i)
    return colors


def plot_tradeoff(
    within_csv: str,
    cross_csv: str,
    out_path: str,
    *,
    dpi: int = 300,
    jitter: float = 0.22,
    scale_mode: str = "auto",
    figsize: Tuple[float, float] = (7.2, 4.5),
    cmap_name: str = "tab10",
    arrow_lw: float = 1.0,
):
    dfw = load_summary(within_csv)
    dfc = load_summary(cross_csv)

    w_auc_cols = auc_columns(dfw)
    c_auc_cols = auc_columns(dfc)

    # best-of-five AUCs
    dfw["AUC_within_best"] = best_of_five(dfw, w_auc_cols, scale_mode=scale_mode)
    dfc["AUC_cross_best"] = best_of_five(dfc, c_auc_cols, scale_mode=scale_mode)

    # merge cross onto within
    df = dfw.merge(
        dfc[["Mask", "AUC_cross_best"]],
        on="Mask",
        how="left",
        validate="one_to_one",
    )

    # Ensure numeric channel counts and drop incomplete rows
    df["N_ch"] = pd.to_numeric(df["N_ch"], errors="coerce")
    df = df.dropna(subset=["N_ch", "AUC_within_best", "AUC_cross_best"]).copy()
    df["N_ch"] = df["N_ch"].astype(int)

    df = df.sort_values(
    by=["N_ch", "AUC_within_best", "Mask"],
    ascending=[True, True, True],
    kind="mergesort"   # stabile
    ).reset_index(drop=True)

    if True:  # keep as a block for clarity
        if REFERENCE_MASK in df["Mask"].values:
            df = df[df["Mask"] != REFERENCE_MASK].reset_index(drop=True)


    # price / cost tier
    price_known, cost_tier = get_price_fields(df)
    df["PriceKnown"] = price_known
    df["CostTier"] = cost_tier

    # colors per form factor
    form_colors = build_form_colors(df, cmap_name=cmap_name)

    # y positions
    y_within = df["AUC_within_best"].to_numpy(dtype=float)
    y_cross = df["AUC_cross_best"].to_numpy(dtype=float)

    # x positions: allocate horizontal space per channel-count group to avoid overlaps.
    # This keeps a single-panel plot but expands dense groups (e.g., 6–8 channels)
    # and compresses sparse gaps (e.g., 12 -> 16 -> 19).
    nch_sorted = sorted(df["N_ch"].dropna().astype(int).unique().tolist())
    counts = df["N_ch"].value_counts().to_dict()

    slot = float(jitter)   # in-group spacing (data units)
    group_gap = 1.2        # gap between channel-count groups (data units)

    centers: Dict[int, float] = {}
    x_cursor = 0.0
    for n in nch_sorted:
        width = (counts.get(n, 1) - 1) * slot
        centers[n] = x_cursor + width / 2.0
        x_cursor += width + group_gap

    x_plot = np.empty(len(df), dtype=float)
    for n in nch_sorted:
        idxs = df.index[df["N_ch"] == n].tolist()
        # deterministic ordering within each group (prevents label reshuffling)
        idxs = sorted(idxs, key=lambda i: (y_within[i], str(df.loc[i, "Mask"])))
        c = len(idxs)
        offsets = (np.arange(c) - (c - 1) / 2.0) * slot
        for k, irow in enumerate(idxs):
            x_plot[irow] = centers[n] + offsets[k]

    x_ticks = [centers[n] for n in nch_sorted]
    x_ticklabels = [str(n) for n in nch_sorted]

    fig, ax = plt.subplots(figsize=figsize)

    # --- vertical dashed separators between N_ch groups ---
    for i in range(len(nch_sorted) - 1):
        n_left = nch_sorted[i]
        n_right = nch_sorted[i + 1]

        # right edge of left group and left edge of right group
        w_left = (counts.get(n_left, 1) - 1) * slot
        w_right = (counts.get(n_right, 1) - 1) * slot

        right_edge_left = centers[n_left] + w_left / 2.0
        left_edge_right = centers[n_right] - w_right / 2.0

        xb = 0.5 * (right_edge_left + left_edge_right)  # boundary in the gap
        ax.axvline(xb, linestyle="--", linewidth=0.6, color="black", alpha=0.28, zorder=0)


    # chance level
    #ax.axhline(50.0, linewidth=1.0, linestyle="--", alpha=0.6)

    # arrows first (so markers stay on top)
    for i in range(len(df)):
        if not np.isfinite(y_within[i]) or not np.isfinite(y_cross[i]):
            continue
        ax.annotate(
            "",
            xy=(x_plot[i], y_cross[i]),
            xytext=(x_plot[i], y_within[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=arrow_lw,
                color="black",
                alpha=0.8,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=10,
            ),
            zorder=1,
        )

    # markers (within only)
    for i in range(len(df)):
        mask = str(df.loc[i, "Mask"])
        form = str(df.loc[i, "Form"])
        color = form_colors.get(form, (0, 0, 0, 1))

        if not np.isfinite(y_within[i]) or not np.isfinite(x_plot[i]):
            continue

        # marker style
        if mask == REFERENCE_MASK:
            marker = "*"
            msize = 260  # reference size
        else:
            marker = "o" if bool(df.loc[i, "PriceKnown"]) else "s"
            tier = int(df.loc[i, "CostTier"])
            msize = COST_TIER_SIZES.get(tier, COST_TIER_SIZES[2])

        ax.scatter(
            [x_plot[i]],
            [y_within[i]],
            s=msize,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidths=0.6,
            alpha=0.65,
            zorder=3,
        )

        # In-marker label (wrapped), centered on the marker
        if mask != REFERENCE_MASK:
            label = wrap_label(pretty_mask_name(mask), width=11)

            rgba = mcolors.to_rgba(color)
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "black"
            outline = "black" if txt_color == "white" else "white"

            # Font size scaled by marker radius (s is marker area in points^2)
            fs = float(np.clip(0.55 * np.sqrt(msize), 9.0, 10.0))

            if marker == "s":
                r_pts = 0.5 * np.sqrt(msize)          # metà lato del quadrato
            else:
                r_pts = np.sqrt(msize / np.pi)        # raggio equivalente del cerchio

            pad_pts = 25  # padding extra sopra il marker (aumenta se vuoi)
            r_pts = r_pts * 0.5
            ax.annotate(
                label,
                xy=(x_plot[i], y_within[i]),          # centro marker
                xytext=(0, r_pts + pad_pts),          # sposta in alto di ~1 raggio
                textcoords="offset points",
                rotation=90,
                rotation_mode="anchor",
                ha="center",
                va="center",                             # <--- testo “parte dall’alto” e va verso il basso
                fontsize=fs,
                #fontweight="bold",
                multialignment="center",
                linespacing=0.85,
                color=txt_color,
                zorder=6,
                path_effects=[pe.withStroke(linewidth=1.0, foreground=outline)],
            )

    # axes

    ax.set_xlabel(r"Channel count")
    ax.set_ylabel("ROC-AUC (%)")

    # x ticks (equispaced) labelled with actual channel counts
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    x_pad = 0.8
    ax.set_xlim(np.nanmin(x_plot) - x_pad, np.nanmax(x_plot) + x_pad)


    # y limits with padding
    y_all = np.concatenate([y_within[np.isfinite(y_within)], y_cross[np.isfinite(y_cross)]])
    if y_all.size > 0:
        lo = max(45.0, float(np.min(y_all)) - 3.0)
        hi = min(90.0, float(np.max(y_all)) + 3.0)
        ax.set_ylim(lo, hi)

    ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)

    # -----------------------------
    # Legends (minimal but complete)
    # -----------------------------

    # form factor legend (color)
    form_handles = []
    for f in FORM_ORDER:
        if f not in form_colors:
            continue
        form_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=form_colors[f],
                markeredgecolor="black",
                markersize=7,
                label=f,
            )
        )
    leg1 = ax.legend(
        handles=form_handles,
        title=None,
        loc="lower right",
        frameon=True,
        borderpad=0.6,
        labelspacing=0.4,
        handletextpad=0.6,
     )
    ax.add_artist(leg1)

    # cost tier legend (size)
    size_handles = []
    for tier in [1, 2, 3, 4]:
        s_area = COST_TIER_SIZES[tier]
        # markersize in legend is radius-ish; use sqrt(area) scaling
        ms = max(4.0, np.sqrt(s_area) / 2.2)
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=ms,
                label=COST_TIER_LABELS[tier],
            )
        )

    # shape legend: price unavailable
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Price unavailable",
        ),
    ]

    #ax.legend(
    #    handles=size_handles + shape_handles,
    #    loc="lower right",
    #    frameon=True,
    #    borderpad=0.6,
    #    labelspacing=0.4,
    #   handletextpad=0.6,
    #)
    # Keep only the relevant axis spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Optional tick placement for visual consistency.
    ax.tick_params(top=False, right=True)

    # save
    out_path = str(Path(out_path))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--within", 
                    default=str(Path(__file__).resolve().parents[1] / "results" / "within_session" / "__summary.csv"), 
                    help="Path to within-session __summary.csv")
    ap.add_argument("--cross", 
                    default=str(Path(__file__).resolve().parents[1] / "results" / "cross_subject" / "__summary_CrossSub.csv"), 
                    help="Path to cross-subject __summary_CrossSub.csv")
    ap.add_argument("--out", 
                    default=str(Path(__file__).resolve().parents[1] / "figures" / "topology_tradeoff.png"), 
                    help="Output image path (png/pdf)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--drop_fullcho", action="store_true", help="Exclude FULL_CHO reference from the plot")
    ap.add_argument("--jitter", type=float, default=1.0, help="Deterministic x-jitter for shared N_ch")
    ap.add_argument("--scale", choices=["auto", "percent", "fraction"], default="auto",
                    help="auto: detect 0..1 vs 0..100; percent: assume already 0..100; fraction: multiply by 100")
    ap.add_argument("--fig_w", type=float, default=14)
    ap.add_argument("--fig_h", type=float, default=8)
    ap.add_argument("--cmap", default="tab10", help="Matplotlib categorical colormap name")
    ap.add_argument("--arrow_lw", type=float, default=1.0)
    args = ap.parse_args()

    plot_tradeoff(
        within_csv=args.within,
        cross_csv=args.cross,
        out_path=args.out,
        dpi=args.dpi,
        jitter=args.jitter,
        scale_mode=args.scale,
        figsize=(args.fig_w, args.fig_h),
        cmap_name=args.cmap,
        arrow_lw=args.arrow_lw,
    )


if __name__ == "__main__":
    main()
