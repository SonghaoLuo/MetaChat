"""
_compass_utils.py
=================
Compute metabolite-level production / consumption scores from Compass
reaction-level penalties and Recon2 reaction metadata.

Adapted from compute_reaction.py (compare_compass/compute_reaction.py).
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) Metabolite name normalisation
# ---------------------------------------------------------------------------

def normalize_met_name(m: str) -> str:
    """Canonicalise metabolite string so that
    ``"XXX [c]"`` and ``"XXX[c]"`` compare equal.

    Rules
    -----
    * Collapse all runs of whitespace to a single space.
    * Remove the space immediately before a compartment tag ``[x]``.
    * Strip leading / trailing whitespace.
    """
    m = " ".join(m.split())
    m = re.sub(r"\s+(\[[a-z]\])", r"\1", m)
    return m.strip()


# ---------------------------------------------------------------------------
# 2) Reaction-formula parser
# ---------------------------------------------------------------------------

_ARROW_RE = re.compile(r"\s*(<-->|-->)\s*")


def _parse_side(side_str: str) -> Dict[str, float]:
    """Parse one side of a reaction formula into {normalised_met: coeff}."""
    result: Dict[str, float] = {}
    if not side_str or not side_str.strip():
        return result

    if side_str.strip() == "\u03c6":
        return result

    tokens = re.split(r"\s*\+\s*(?=\d)", side_str)

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^([\d.]+)\s*\*\s*(.+)$", tok)
        if m:
            coeff = float(m.group(1))
            met = normalize_met_name(m.group(2))
        else:
            coeff = 1.0
            met = normalize_met_name(tok)
        result[met] = result.get(met, 0.0) + coeff
    return result


def parse_rxn_formula(formula: str) -> Dict[str, float]:
    """Parse a reaction formula string into
    ``{normalised_metabolite: signed_coefficient}``.

    Reactants (left of arrow) get **negative** coefficients.
    Products  (right of arrow) get **positive** coefficients.
    """
    formula = formula.split("\n")[0].strip()

    arrow_match = _ARROW_RE.search(formula)
    if arrow_match is None:
        warnings.warn(f"No arrow found in formula: {formula!r}")
        return {}

    lhs = formula[:arrow_match.start()]
    rhs = formula[arrow_match.end():]

    stoich: Dict[str, float] = {}
    for met, coeff in _parse_side(lhs).items():
        stoich[met] = stoich.get(met, 0.0) - coeff
    for met, coeff in _parse_side(rhs).items():
        stoich[met] = stoich.get(met, 0.0) + coeff
    return stoich


# ---------------------------------------------------------------------------
# 3) Internal-reaction filter
# ---------------------------------------------------------------------------

_COMPARTMENT_RE = re.compile(r"\[([a-z])\]")


def _extract_compartments(formula: str) -> set:
    """Return the set of compartment letters found in *formula*."""
    return set(_COMPARTMENT_RE.findall(formula.split("\n")[0]))


def is_internal_reaction(
    formula: str,
    target_compartment: str,
    *,
    exclude_transport: bool = True,
    exclude_boundary: bool = True,
) -> bool:
    """Decide whether a reaction formula is *internal* w.r.t.
    *target_compartment*.

    Parameters
    ----------
    formula : str
        The ``rxn_formula`` string.
    target_compartment : str
        Single-letter compartment, e.g. ``"c"``.
    exclude_transport : bool
        Drop reactions that span >1 compartment.
    exclude_boundary : bool
        Drop reactions where one side has zero species or the formula
        contains the empty-set symbol ``\u03c6``.
    """
    line = formula.split("\n")[0].strip()
    comps = _extract_compartments(line)

    if target_compartment not in comps:
        return False

    if exclude_transport and len(comps) > 1:
        return False

    if exclude_boundary:
        if "\u03c6" in line:
            return False
        arrow_match = _ARROW_RE.search(line)
        if arrow_match:
            lhs = line[:arrow_match.start()].strip()
            rhs = line[arrow_match.end():].strip()
            if len(_parse_side(lhs)) == 0 or len(_parse_side(rhs)) == 0:
                return False

    return True


# ---------------------------------------------------------------------------
# 4) Main scoring function
# ---------------------------------------------------------------------------

def compute_metabolite_scores(
    rxn_md_df: pd.DataFrame,
    compass_rxn_mat: pd.DataFrame,
    metabolite: str,
    *,
    input_is_penalty: bool = True,
    assume_neg_is_reverse: bool = True,
    weight_by_stoich: bool = True,
    norm: str = "rank",
    agg: str = "topk_mean",
    topk: int = 5,
    exclude_exchange: bool = True,
    exclude_sink_demand: bool = True,
    exclude_transport: bool = True,
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, dict]:
    """Compute per-cell metabolite Production / Consumption scores.

    Parameters
    ----------
    rxn_md_df : DataFrame
        Loaded from ``rxn_md.csv`` (index = rxn_code_nodirection).
        Must contain column ``rxn_formula``.
    compass_rxn_mat : DataFrame
        Compass reaction result matrix.  Index = directional IDs
        (``<rxn>_pos`` / ``<rxn>_neg``), columns = cell / spot IDs,
        values = penalty (or score, see *input_is_penalty*).
    metabolite : str
        Metabolite with compartment tag, e.g. ``"L-glutamate(1-) [c]"``.
    input_is_penalty : bool
        If True (default), convert penalty to score via
        ``score = -log1p(penalty)``.
    assume_neg_is_reverse : bool
        If True (default), the ``_neg`` direction is treated as the
        *reverse* of the formula: stoichiometric signs are flipped.
    weight_by_stoich : bool
        Weight reactions by ``|stoichiometric coefficient|`` during
        aggregation.
    norm : str
        Per-reaction normalisation across cells.  ``"rank"`` (default)
        uses percentile rank; ``"minmax"`` uses min-max scaling.
    agg : str
        ``"topk_mean"`` (default) or ``"max"``.
    topk : int
        Number of top reactions used when ``agg="topk_mean"``.
    exclude_exchange : bool
        Drop reactions whose ID starts with ``"EX_"``, ``"DM_"``, or
        ``"sink_"``.
    exclude_sink_demand : bool
        Drop boundary reactions (one-sided formulas, ``\u03c6`` symbol).
    exclude_transport : bool
        Drop reactions spanning multiple compartments.
    eps : float
        Threshold for treating a reaction row as near-constant.

    Returns
    -------
    scores_df : DataFrame
        Index = cell/spot IDs, columns = ``["Prod", "Cons", "Balance"]``.
    details : dict
        Keys: ``producing_rxns``, ``consuming_rxns``, ``dropped_rxns``,
        ``all_internal_rxns``, ``stoich_map``.
    """
    met_norm = normalize_met_name(metabolite)

    comp_match = re.search(r"\[([a-z])\]$", met_norm)
    if comp_match is None:
        raise ValueError(
            f"Cannot determine compartment from metabolite: {metabolite!r}. "
            "Expected trailing [x] tag."
        )
    target_comp = comp_match.group(1)

    # Step 1: fast pre-filter
    met_substr = met_norm.replace("[", " [")
    met_substr2 = met_norm

    mask = rxn_md_df["rxn_formula"].fillna("").apply(
        lambda f: met_substr in f or met_substr2 in f
    )
    candidates = rxn_md_df.loc[mask].copy()

    # Step 2: exclude exchange / demand / sink by rxn_code prefix
    if exclude_exchange:
        idx = candidates.index.astype(str)
        keep = ~(
            idx.str.startswith("EX_")
            | idx.str.startswith("DM_")
            | idx.str.startswith("sink_")
        )
        candidates = candidates.loc[keep]

    # Step 3: parse formula, check internal, extract stoichiometry
    stoich_map: Dict[str, float] = {}
    dropped_not_internal: List[str] = []

    for rxn_code, row in candidates.iterrows():
        formula = str(row["rxn_formula"])
        if not is_internal_reaction(
            formula,
            target_comp,
            exclude_transport=exclude_transport,
            exclude_boundary=exclude_sink_demand,
        ):
            dropped_not_internal.append(str(rxn_code))
            continue

        parsed = parse_rxn_formula(formula)
        if met_norm in parsed:
            stoich_map[str(rxn_code)] = parsed[met_norm]

    if not stoich_map:
        warnings.warn(
            f"No internal reactions found for metabolite {metabolite!r}."
        )
        empty_scores = pd.DataFrame(
            0.0,
            index=compass_rxn_mat.columns,
            columns=["Prod", "Cons", "Balance"],
        )
        return empty_scores, {
            "producing_rxns": [],
            "consuming_rxns": [],
            "dropped_rxns": dropped_not_internal,
            "all_internal_rxns": [],
            "stoich_map": {},
        }

    # Step 4: map to directional Compass IDs and build per-cell matrix
    compass_idx_set = set(compass_rxn_mat.index)
    cells = compass_rxn_mat.columns

    producing_ids: List[str] = []
    consuming_ids: List[str] = []
    weights: Dict[str, float] = {}
    dropped_missing: List[str] = []
    dropped_constant: List[str] = []
    rows_to_use: Dict[str, pd.Series] = {}

    for rxn_code, coeff_fwd in stoich_map.items():
        for suffix in ("_pos", "_neg"):
            dir_id = f"{rxn_code}{suffix}"
            if dir_id not in compass_idx_set:
                continue

            raw = compass_rxn_mat.loc[dir_id]

            if suffix == "_neg" and assume_neg_is_reverse:
                eff_coeff = -coeff_fwd
            else:
                eff_coeff = coeff_fwd

            if input_is_penalty:
                vals = -np.log1p(raw.values.astype(float))
            else:
                vals = raw.values.astype(float)

            series = pd.Series(vals, index=cells)

            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            iqr = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
            if (vmax - vmin) < eps and iqr < eps:
                dropped_constant.append(dir_id)
                continue

            if norm == "minmax":
                normed = (series - vmin) / (vmax - vmin)
            else:
                normed = series.rank(pct=True, method="average")
            rows_to_use[dir_id] = normed

            w = abs(eff_coeff) if weight_by_stoich else 1.0
            weights[dir_id] = w

            if eff_coeff > 0:
                producing_ids.append(dir_id)
            elif eff_coeff < 0:
                consuming_ids.append(dir_id)
            else:
                dropped_constant.append(dir_id)

        pos_exists = f"{rxn_code}_pos" in compass_idx_set
        neg_exists = f"{rxn_code}_neg" in compass_idx_set
        if not pos_exists and not neg_exists:
            dropped_missing.append(rxn_code)

    # Step 5: aggregate into Prod / Cons per cell
    def _aggregate(ids: List[str]) -> pd.Series:
        if not ids:
            return pd.Series(np.nan, index=cells)

        arr = np.column_stack([rows_to_use[rid].values for rid in ids])
        w_arr = np.array([weights[rid] for rid in ids])

        if agg == "max":
            return pd.Series(np.nanmax(arr, axis=1), index=cells)

        k = min(topk, len(ids))
        n_rxns = arr.shape[1]

        if n_rxns <= k:
            result = np.nansum(arr * w_arr, axis=1) / np.nansum(
                np.where(np.isnan(arr), 0.0, 1.0) * w_arr, axis=1
            )
        else:
            arr_filled = np.where(np.isnan(arr), -np.inf, arr)
            top_idx = np.argpartition(-arr_filled, k, axis=1)[:, :k]
            row_ix = np.arange(arr.shape[0])[:, None]
            top_vals = arr[row_ix, top_idx]
            top_ws = w_arr[top_idx]
            valid_mask = ~np.isnan(top_vals)
            wsum = np.sum(np.where(valid_mask, top_vals * top_ws, 0.0), axis=1)
            wden = np.sum(np.where(valid_mask, top_ws, 0.0), axis=1)
            result = np.where(wden > 0, wsum / wden, np.nan)

        return pd.Series(result, index=cells)

    prod = _aggregate(producing_ids)
    cons = _aggregate(consuming_ids)

    prod_filled = prod.fillna(0.0)
    cons_filled = cons.fillna(0.0)

    scores_df = pd.DataFrame({
        "Prod": prod_filled,
        "Cons": cons_filled,
        "Balance": prod_filled - cons_filled,
    }, index=cells)

    details = {
        "producing_rxns": producing_ids,
        "consuming_rxns": consuming_ids,
        "dropped_rxns": dropped_not_internal + dropped_missing + dropped_constant,
        "all_internal_rxns": list(stoich_map.keys()),
        "stoich_map": stoich_map,
    }

    return scores_df, details
