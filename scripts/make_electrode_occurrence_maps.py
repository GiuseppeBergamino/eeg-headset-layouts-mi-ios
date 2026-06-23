#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import mne
import inspect
from collections import Counter

from headset_profiles import HEADSETS, normalize_channel_names

# ============================================================
# CONFIGURATION
# ============================================================
FORMS = ["Cap", "Helmet", "Headband", "Headphones"]
EXCLUDE_KEYS = {"FULL_CHO"}                 # do not count the reference cap
DROP_RAW_CHANNELS = {"FT9", "FT10", "TP9", "TP10"}  # excluded before channel-name normalization

MONTAGE_NAME = "standard_1005"              # richer than standard_1020
OUTLINES = "head"                           # compatible with MNE 1.11
EXTRAPOLATE = "local"

# marker style
MARKER_SIZE_PT2 = 320                       # area in pt^2
MARKER_EDGE_COLOR = "black"
MARKER_EDGE_LW = 0.5

# labels
LABEL_FONTSIZE = 7.5
LABEL_WEIGHT = "bold"
ANNOTATE_THRESHOLD = 1                      # increase to show fewer labels

# head scaling
SPHERE_RADIUS_SCALE = 1.35                  # increase if lateral electrodes fall outside the outline
AX_LIM_SCALE = 1.05 #0.88                         # <1 enlarges the head within each panel

# quantized colormap: 1 = white, max = blue
VMAX = None                                 # None = auto (max globale)
N_LEVELS = None                             # None = auto (tutti i valori interi)
CMAP_NAME = "Blues"

# layout / export
FIGSIZE = (7.6, 7.6)                   # quadrata, più adatta a colonna singola
CBAR_POS = (0.12, 0.06, 0.76, 0.03)   # (left, bottom, width, height) in figure coords
SUBPLOT_BOTTOM = 0.07                # spazio riservato sotto per la colorbar

OUT_PNG = "electrode_popularity_by_form_color.png"
DPI = 300
# ============================================================


MONTAGE = mne.channels.make_standard_montage(MONTAGE_NAME)


def _iter_channels(profile):
    for ch in getattr(profile, "channels", []):
        if ch in DROP_RAW_CHANNELS:
            continue
        yield normalize_channel_names(ch)


def count_global() -> Counter:
    c = Counter()
    for key, hs in HEADSETS.items():
        if key in EXCLUDE_KEYS:
            continue
        for ch in _iter_channels(hs):
            c[ch] += 1
    return c


def count_by_form(form: str) -> Counter:
    c = Counter()
    for key, hs in HEADSETS.items():
        if key in EXCLUDE_KEYS:
            continue
        if getattr(hs, "form_factor", None) != form:
            continue
        for ch in _iter_channels(hs):
            c[ch] += 1
    return c


def make_union_info(union_ch):
    info = mne.create_info(ch_names=union_ch, sfreq=100.0, ch_types="eeg")
    info.set_montage(MONTAGE, on_missing="ignore")
    return info


def fit_fixed_sphere(info_union, radius_scale=1.10):
    """Sphere (x,y,z,r) fissa per tutte le subplot, con raggio scalato."""
    try:
        fit = mne.bem.fit_sphere_to_headshape(info_union, dig_kinds="eeg", units="m")
        if isinstance(fit, tuple) and len(fit) == 2:
            r0, radius = fit
        elif isinstance(fit, dict):
            r0, radius = fit["r0"], fit["radius"]
        else:
            r0, radius = fit.r0, fit.radius
        r0 = np.asarray(r0, float)
        radius = float(radius) * float(radius_scale)
        return (float(r0[0]), float(r0[1]), float(r0[2]), radius)
    except Exception:
        # fallback: stima semplice dalla nuvola 3D
        pos = info_union.get_montage().get_positions()["ch_pos"]
        pts = np.array([pos[ch] for ch in info_union.ch_names if ch in pos], float)
        r0 = pts.mean(axis=0)
        radius = np.max(np.linalg.norm(pts - r0, axis=1)) * float(radius_scale)
        return (float(r0[0]), float(r0[1]), float(r0[2]), float(radius))


def get_topomap_coords(info, sphere):
    """2D coords coerenti con MNE topomap (fallback tra import interni)."""
    try:
        from mne.channels.layout import _find_topomap_coords as find_coords
    except Exception:
        from mne.viz.topomap import _find_topomap_coords as find_coords  # pragma: no cover
    picks = np.arange(len(info.ch_names))
    return find_coords(info, picks=picks, sphere=sphere)


def build_discrete_cmap(vmin, vmax, n_levels=None, cmap_name="Blues"):
    if n_levels is None:
        n_levels = int(vmax - vmin + 1)

    # discrete levels
    base = mpl.cm.get_cmap(cmap_name, n_levels)
    colors = base(np.linspace(0.25, 0.95, n_levels)) # adjust lower bound to tune color gradient
    colors[0] = np.array([1, 1, 1, 1])  # 1 -> bianco puro
    cmap = mpl.colors.ListedColormap(colors)

    # half-integer boundaries for integer values
    bounds = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds


def draw_head(ax, data, info, sphere):
    # Draw only the MNE head outline, without sensors or topomap values.
    kwargs = dict(
        axes=ax,
        show=False,
        contours=0,
        sensors=False,
        sphere=sphere,
        outlines=OUTLINES,
        extrapolate=EXTRAPOLATE,
    )
    sig = inspect.signature(mne.viz.plot_topomap)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    im, _ = mne.viz.plot_topomap(data, info, **kwargs)
    if im is not None:
        im.set_alpha(0.0)


def plot_panel(ax, title, counts, union_ch, coords2d, cmap, norm, lim):
    #ax.set_title(title, fontsize= 8, pad=1)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")

    if not counts:
        return

    # channels available in the montage
    chs = [ch for ch in counts.keys() if ch in union_ch]
    if not chs:
        return

    # indice canale -> coordinata 2D
    idx = [union_ch.index(ch) for ch in chs]
    xy = coords2d[idx]
    vals = np.array([counts[ch] for ch in chs], float)

    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        s=MARKER_SIZE_PT2,
        c=vals,
        cmap=cmap,
        norm=norm,
        edgecolors=MARKER_EDGE_COLOR,
        linewidths=MARKER_EDGE_LW,
        zorder=5,
    )

    # labels centrati + stroke per leggibilità su blu scuro
    vmax_here = float(np.max(vals)) if len(vals) else 1.0
    for (x, y), ch, v in zip(xy, chs, vals):
        if v < ANNOTATE_THRESHOLD:
            continue
        # euristica: se colore scuro -> testo bianco, altrimenti nero
        is_dark = (v / max(vmax_here, 1.0)) > 0.55
        fg = "white" if is_dark else "black"
        stroke = "black" if is_dark else "white"
        t = ax.text(
            x, y, ch,
            ha="center", va="center",
            fontsize=LABEL_FONTSIZE,
            fontweight=LABEL_WEIGHT,
            color=fg,
            zorder=6,
        )
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=stroke)])

    return sc


def main():
    global_counts = count_global()
    # keep only channels available in the montage
    union_ch = sorted([ch for ch in global_counts.keys() if ch in MONTAGE.ch_names])

    info_union = make_union_info(union_ch)
    sphere = fit_fixed_sphere(info_union, radius_scale=SPHERE_RADIUS_SCALE)

    # coordinate 2D coerenti con MNE topomap
    coords2d = get_topomap_coords(info_union, sphere=sphere)

    # lim uguale per tutte, derivato dalle coords (così riempi bene lo spazio)
    base_lim = float(np.max(np.abs(coords2d))) * 1.05
    lim = base_lim * float(AX_LIM_SCALE)

    vmin = 1
    vmax = int(max(global_counts.values())) if global_counts else 1
    if VMAX is not None:
        vmax = int(VMAX)

    cmap, norm, bounds = build_discrete_cmap(vmin, vmax, n_levels=N_LEVELS, cmap_name=CMAP_NAME)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    axes = axes.ravel()

    # disegna testa una volta per panel usando union (poi scatter sopra)
    dummy = np.zeros(len(union_ch), float)

    for form, ax in zip(FORMS, axes):
        draw_head(ax, dummy, info_union, sphere=sphere)   # MNE head outline
        counts = count_by_form(form)
        # keep only channels available in the montage
        counts = Counter({ch: c for ch, c in counts.items() if ch in union_ch})
        plot_panel(ax, form, counts, union_ch, coords2d, cmap, norm, lim)

    # discrete colorbar
    # horizontal discrete colorbar
    cax = fig.add_axes(CBAR_POS)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        cax=cax,
        boundaries=bounds,
        spacing="proportional",
        orientation="horizontal",
    )

    cb.set_label("Occurrences", rotation=0, labelpad=6)

    mid = int(np.ceil((vmin + vmax) / 2))
    cb.set_ticks([vmin, mid, vmax])
    cb.set_ticklabels([str(vmin), str(mid), str(vmax)])

    # place label on top and ticks below for cleaner layout
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")

    fig.subplots_adjust(
        left=0.02, right=0.98,
        top=0.97, bottom=SUBPLOT_BOTTOM,
        wspace=0.01, hspace=0.01
    )

    fig.savefig(OUT_PNG, dpi=DPI)
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
