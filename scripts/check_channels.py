from __future__ import annotations
from typing import Dict, List, Tuple, Optional

from headset_profiles import HEADSETS, normalize_channel_names


def canonicalize_case(ch: str) -> str:
    """
    Tries to canonicalize common 10-20/10-10 casing:
    - Fpz (not FPZ), Iz (not IZ)
    - z is lowercase
    Keeps other letters as in common conventions (AF, FC, CP, PO, FT, TP).
    """
    ch = ch.strip()
    if not ch:
        return ch

    up = ch.upper()

    # handle known mixed-case prefixes
    for pref in ["FP", "AF", "FC", "CP", "PO", "FT", "TP"]:
        if up.startswith(pref):
            rest = up[len(pref):]
            # special: Fp prefix is "Fp", others stay uppercase
            fixed_pref = "Fp" if pref == "FP" else pref
            # force trailing Z -> z
            if rest.endswith("Z"):
                rest = rest[:-1] + "z"
            return fixed_pref + rest

    # handle IZ -> Iz
    if up.startswith("IZ"):
        rest = up[2:]
        if rest.endswith("Z"):
            rest = rest[:-1] + "z"
        return "Iz" + rest

    # single-letter prefixes (F,C,P,O,T) keep uppercase; trailing Z -> z
    if up.endswith("Z"):
        up = up[:-1] + "z"
    return up


def norm(ch: str) -> str:
    """Strip + canonicalize + apply your legacy map."""
    return normalize_channel_names(canonicalize_case(ch))


def suggest_aliases_for_missing(orig: str, full_norm_set: set) -> List[str]:
    """
    Returns a list of suggested aliases (present in FULL_CHO) for a missing channel.
    Suggestions are ONLY returned if the candidate exists in the FULL_CHO set.
    """
    o = orig.strip()
    suggestions: List[str] = []

    # 1) try canonicalize + legacy map (in case the missing was due to casing/spacing)
    c1 = norm(o)
    if c1 in full_norm_set:
        suggestions.append(c1)

    # 2) common typo: '0' used instead of 'O' (e.g., P07 -> PO7)
    if "0" in o:
        c2 = norm(o.replace("0", "O"))
        if c2 in full_norm_set and c2 not in suggestions:
            suggestions.append(c2)

    # 3) common typo: O8 -> PO8 (O8 not standard in 10-10, PO8 is)
    # only if it looks like O<digit>
    if len(o) == 2 and o[0].upper() == "O" and o[1].isdigit():
        c3 = norm("PO" + o[1])
        if c3 in full_norm_set and c3 not in suggestions:
            suggestions.append(c3)

    # 4) handle "TP9/TP10" style already covered by legacy_map, but keep for safety
    if o.upper() in {"TP9", "TP10"}:
        c4 = norm(o.upper())
        if c4 in full_norm_set and c4 not in suggestions:
            suggestions.append(c4)

    return suggestions


def check_headsets_vs_full_cho(
    headsets: Dict[str, object],
    full_key: str = "FULL_CHO",
    show_only_missing: bool = True,
    max_names_len: int = 150,
) -> None:
    if full_key not in headsets:
        raise KeyError(f"'{full_key}' not found in HEADSETS.")

    full = headsets[full_key]
    full_ch = getattr(full, "channels", [])
    full_norm_set = {norm(ch) for ch in full_ch}

    rows: List[Tuple[str, int, str]] = []
    totals = 0
    missing_count = 0

    for key, hs in headsets.items():
        if key == full_key:
            continue

        totals += 1
        hs_name = getattr(hs, "name", key)
        hs_ch = list(getattr(hs, "channels", []))

        # compute missing after normalization
        hs_norm = [norm(ch) for ch in hs_ch]
        missing_orig = [orig for orig, n in zip(hs_ch, hs_norm) if n not in full_norm_set]

        if show_only_missing and len(missing_orig) == 0:
            continue

        if len(missing_orig) > 0:
            missing_count += 1

        # Build display string with alias hints:
        # - if alias exists: orig→alias
        # - else: orig×
        parts = []
        for m in missing_orig:
            sugg = suggest_aliases_for_missing(m, full_norm_set)
            if sugg:
                # if multiple, show first and keep others in parentheses
                if len(sugg) == 1:
                    parts.append(f"{m}→{sugg[0]}")
                else:
                    parts.append(f"{m}→{sugg[0]}(+{','.join(sugg[1:])})")
            else:
                parts.append(f"{m}×")

        miss_str = ", ".join(parts)
        if len(miss_str) > max_names_len:
            miss_str = miss_str[: max_names_len - 3] + "..."

        rows.append((hs_name, len(missing_orig), miss_str))

    # sort: most missing first
    rows.sort(key=lambda r: (-r[1], r[0].lower()))

    # print table
    print("\nHeadset channels vs FULL_CHO missing channels report")
    col1 = "headset_name"
    col2 = "missing channels"
    col3 = "missing channel names"

    w1 = max(len(col1), max((len(r[0]) for r in rows), default=0))
    w2 = max(len(col2), max((len(str(r[1])) for r in rows), default=0))

    sep = f"{'-'*w1}-+-{'-'*w2}-+-{'-'*max(len(col3), 30)}"
    print(f"{col1:<{w1}} | {col2:>{w2}} | {col3}")
    print(sep)
    for name, nmiss, miss in rows:
        print(f"{name:<{w1}} | {nmiss:>{w2}} | {miss}")

    print(sep)
    print(f"Checked headsets: {totals} | With missing channels: {missing_count} | Full match: {totals - missing_count}")


if __name__ == "__main__":
    check_headsets_vs_full_cho(HEADSETS, full_key="FULL_CHO", show_only_missing=True)
