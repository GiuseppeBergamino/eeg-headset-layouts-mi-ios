import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FORM_ORDER = ["Cap", "Helmet", "Headband", "Headphones"]

# Mapping from internal headset codes to display names
HEADSET_NAME_MAP = {
    "FULL_CHO": "Reference Cho2017",
    "BITBRAIN_DIADEM": "Bitbrain Diadem",
    "BITBRAIN_HERO": "Bitbrain Hero",
    "BITBRAIN_AIR": "Bitbrain Air",
    "BITBRAIN_IKON": "Bitbrain Ikon",
    "BRAINBIT_DRAGON": "BrainBit Dragon",
    "BRAINBIT_HEADBAND_PRO": "BrainBit Headband Pro",
    "BRAINBIT_HEADBAND": "BrainBit Headband",
    "BRAINBIT_HEADPHONES": "BrainBit Headphones",
    "CGX_QUICK_32R": "CGX Quick 32R",
    "CGX_QUICK_20R": "CGX Quick 20R",
    "EMOTIV_FLEX": "Emotiv Flex 2",
    "EMOTIV_EPOCH_X": "Emotiv EPOC X",
    "EMOTIV_INSIGHT": "Emotiv Insight",
    "EMOTIV_MN8": "Emotiv MN8",
    "GTEC_UNICORN_HYBRID": "g.tec Unicorn Hybrid",
    "MUSE_2_HEADBAND": "Muse 2 Headband",
    "MUSE_S_ATHENA": "Muse S Athena",
    "MINDROVE_VISION": "Mindrove Vision",
    "MINDROVE_ARC": "Mindrove Arc",
    "MINDROVE_LUCID": "Mindrove Lucid",
    "MINDROVE_BRIGHT": "Mindrove Bright",
    "NEEURO_SENZEBAND": "Neeuro SenzeBand",
    "NEUROSITY_CROWN": "Neurosity Crown",
    "NEUROSKY_MINDWAVE_2": "Neurosky Mindwave 2",
    "ULTRACORTEX_16": "Ultracortex IV",
    "ULTRACORTEX_8": "Ultracortex IV",
    "WS_DSI_24": "W. S. DSI-24",
    "WS_DSI_7": "W. S. DSI-7",
    "WS_DSI_VR300": "W. S.  DSI VR300",
    "WS_DSI_VRVEP": "W. S. DSI VRVEP",
    "ABM_B_ALERT_X24" : "A. B. M. B-Alert X24",
    "ABM_B_ALERT_X10" : "A. B. M. B-Alert X10",
}

PIPELINE_NAME_MAP = {
    "CSP_LDA": "CSP+LDA",
    "CSP_SVM": "CSP+SVM",
    "TS_SVM": "TS+SVM",
    "TS_LR": "TS+LR",
    "TS_EL": "TS+EL",
}

PM_RE = re.compile(
    r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:±|\+/-|\+-)\s*([0-9]+(?:[.,][0-9]+)?)\s*$"
)


def parse_mean_std(cell: str) -> tuple[float, float]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return (np.nan, np.nan)
    s = str(cell).strip()
    m = PM_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse mean±std from: {cell!r}")
    mean_s = m.group(1).replace(",", ".")
    std_s = m.group(2).replace(",", ".")
    return float(mean_s), float(std_s)


def pretty_mask_name(mask: str) -> str:
    return HEADSET_NAME_MAP.get(mask, mask)


def load_and_prepare(csv_path: str, order_nch: str = "desc") -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)

    required = {"Mask", "Form", "N_ch"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}; found: {list(df.columns)}")

    auc_cols = [c for c in df.columns if c.endswith("_AUC")]
    if not auc_cols:
        raise ValueError("No AUC columns found (expected columns ending with '_AUC').")

    pipelines = [c[:-4] for c in auc_cols]  # remove _AUC

    # ---- Force pipeline order ----
    preferred = ["CSP_LDA", "CSP_SVM", "TS_SVM", "TS_LR", "TS_EL"]
    pipelines = [p for p in preferred if p in pipelines] + [p for p in pipelines if p not in preferred]
# ------------------------------------------------------

    df = df.copy()
    df["N_ch"] = pd.to_numeric(df["N_ch"], errors="coerce")

    df["Form"] = pd.Categorical(df["Form"], categories=FORM_ORDER, ordered=True)

    asc_nch = (order_nch.lower() == "asc")
    df = df.sort_values(
        by=["Form", "N_ch", "Mask"],
        ascending=[True, asc_nch, True],
        kind="mergesort",
    ).reset_index(drop=True)

    df = df.drop_duplicates(subset=["Mask"], keep="first").reset_index(drop=True)

    return df, pipelines


def build_matrices(df: pd.DataFrame, pipelines: list[str], scale: str = "percent") -> tuple[np.ndarray, np.ndarray]:
    n_headsets = df.shape[0]
    n_pipes = len(pipelines)

    means = np.full((n_headsets, n_pipes), np.nan, dtype=float)
    stds = np.full((n_headsets, n_pipes), np.nan, dtype=float)

    for j, pipe in enumerate(pipelines):
        col = f"{pipe}_AUC"
        parsed = df[col].apply(parse_mean_std)
        means[:, j] = parsed.apply(lambda t: t[0]).to_numpy()
        stds[:, j] = parsed.apply(lambda t: t[1]).to_numpy()

    if scale == "fraction":
        means = means / 100.0
        stds = stds / 100.0

    return means, stds


def plot_heatmap(
    df: pd.DataFrame,
    pipelines: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    out_path: str,
    *,
    scale: str = "percent",
    cmap: str = "viridis",
    annotate: bool = True,
    annot_fs: float = 9.0,
    pipeline_fs: float = 10.0,
    rowlabel_fs: float = 9.0,
    vmax_fullcho: bool = False,
    dpi: int = 300,
):
    n_headsets, n_pipes = means.shape

    data = np.ma.masked_invalid(means)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    vmin = np.nanmin(means)
    if vmax_fullcho and (df["Mask"] == "FULL_CHO").any():
        full_idx = int(df.index[df["Mask"] == "FULL_CHO"][0])
        vmax = float(np.nanmax(means[full_idx, :]))
    else:
        vmax = np.nanmax(means)

    # Figure size suitable for a two-column layout.
    fig_w = 0.95 * n_pipes + 7.0
    fig_h = 0.30 * n_headsets + 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap_obj)

    # Pipeline labels on top.
    pretty_pipes = [PIPELINE_NAME_MAP.get(p, p) for p in pipelines]
    ax.set_xticks(np.arange(n_pipes))
    ax.set_xticklabels(pretty_pipes, rotation=0, ha="center", fontsize=pipeline_fs, fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)

    # Y labels: nomi + (N_ch)
    nch_series = pd.to_numeric(df["N_ch"], errors="coerce")
    row_labels = []
    for mask, nch in zip(df["Mask"].astype(str), nch_series):
        disp = pretty_mask_name(mask)
        if pd.notna(nch):
            row_labels.append(f"{disp} ({int(nch)} ch)")
        else:
            row_labels.append(f"{disp} (-)")

    ax.set_yticks(np.arange(n_headsets))
    ax.tick_params(axis="y", pad=4)  # distance between headset names and the heatmap

    ax.set_yticklabels(row_labels, fontsize=rowlabel_fs)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, n_pipes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_headsets, 1), minor=True)
    ax.grid(which="minor", linewidth=0.35)
    ax.tick_params(which="minor", left=False, bottom=False)

    # ---- Brackets and form-factor labels to the left of row names ----
    # Use the Y-axis transform: x in axis coordinates, y in data coordinates.
    trans = ax.get_yaxis_transform()

    forms = df["Form"].astype(str).to_numpy()
    bounds = []
    start = 0
    for i in range(1, len(forms) + 1):
        if i == len(forms) or forms[i] != forms[i - 1]:
            bounds.append((forms[i - 1], start, i - 1))
            start = i

    # Position in axis coordinates: 0 is the left edge of the heatmap axis.
    # Slightly negative values place brackets outside the axis but inside the figure.
    FORM_X = -0.235 # distance between group vertical line and heatmap
    TICK = 0.15 # length of horizontal bracket ticks
    TEXT_X = FORM_X - 0.01 # distance of group labels from the heatmap

    for form, y0, y1 in bounds:
        ax.plot([FORM_X, FORM_X], [y0 - 0.5, y1 + 0.5],
                transform=trans, color="black", linewidth=1.2, clip_on=False)
        ax.plot([FORM_X, FORM_X + TICK], [y0 - 0.5, y0 - 0.5],
                transform=trans, color="black", linewidth=1.2, clip_on=False)
        ax.plot([FORM_X, FORM_X + TICK], [y1 + 0.5, y1 + 0.5],
                transform=trans, color="black", linewidth=1.2, clip_on=False)

        ax.text(TEXT_X, (y0 + y1) / 2, form,
                transform=trans, rotation=90,
                va="center", ha="center",
                fontsize=rowlabel_fs, fontweight="bold",
                clip_on=False)

    # Cell annotations.
    if annotate:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for i in range(n_headsets):
            for j in range(n_pipes):
                m = means[i, j]
                s = stds[i, j]
                if np.isnan(m):
                    continue

                rgba = im.cmap(norm(m))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "black" if luminance > 0.6 else "white"

                if scale == "percent":
                    txt = f"{m:.2f}±{s:.2f}"
                else:
                    txt = f"{m:.3f}±{s:.3f}"

                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=annot_fs, color=color, fontweight="bold")

    # Colorbar
    #cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    #cbar.set_label("ROC-AUC (%)" if scale == "percent" else "ROC-AUC")

    # Margins: leave room on the left for form-factor labels and brackets.
    LEFT_MARGIN = 0.26
    fig.subplots_adjust(left=0.2, right=0.99, top=0.97, bottom=0.01) # figure centering



    # Save without aggressive cropping.
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[1] / "results" / "within_session" / "__summary.csv"),
        help="Path to __summary.csv",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "figures" / "within_heatmap.png"),
        help="Output path (pdf/png)",
    )
    ap.add_argument("--order_nch", choices=["asc", "desc"], default="desc", help="Sort N_ch within each Form")
    ap.add_argument("--scale", choices=["percent", "fraction"], default="percent",
                    help="percent keeps e.g. 70.56; fraction divides by 100 -> 0.7056")
    ap.add_argument("--vmax_fullcho", action="store_true",
                    help="Set vmax using FULL_CHO row (max across pipelines) if present")
    ap.add_argument("--no_annot", action="store_true", help="Disable cell annotations")
    ap.add_argument("--annot_fs", type=float, default=14.0, help="Annotation font size (numbers inside cells)")
    ap.add_argument("--pipeline_fs", type=float, default=14.0, help="Pipeline label font size (top)")
    ap.add_argument("--rowlabel_fs", type=float, default=10.0, help="Row label font size (headset names)")
    ap.add_argument("--dpi", type=int, default=300, help="Save DPI (useful for PNG)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    args = ap.parse_args()

    df, pipelines = load_and_prepare(args.csv, order_nch=args.order_nch)
    # ---- FORCE PIPELINE ORDER (TS_SVM before TS_LR) ----
    PIPE_ORDER = ["CSP+LDA", "CSP+SVM", "TS+SVM", "TS+LR", "TS+EL"]
    pipelines = [p for p in PIPE_ORDER if p in pipelines] + [p for p in pipelines if p not in PIPE_ORDER]
    print("Pipeline order used:", pipelines)

    means, stds = build_matrices(df, pipelines, scale=args.scale)

    plot_heatmap(
        df=df,
        pipelines=pipelines,
        means=means,
        stds=stds,
        out_path=str(Path(args.out)),
        scale=args.scale,
        cmap=args.cmap,
        annotate=(not args.no_annot),
        annot_fs=args.annot_fs,
        pipeline_fs=args.pipeline_fs,
        rowlabel_fs=args.rowlabel_fs,
        vmax_fullcho=args.vmax_fullcho,
        dpi=args.dpi,
    )

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
