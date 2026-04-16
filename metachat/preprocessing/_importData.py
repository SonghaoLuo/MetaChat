import io
import pkgutil
import anndata
import numpy as np
import pandas as pd
from typing import Optional

def MetaChatDB(
    species = "mouse"
):
    """
    Extract metabolite-sensor pairs from MetaChatDB.

    Parameters
    ----------
    species
        The species of the ligand-receptor pairs. Choose between 'mouse' and 'human'.

    Returns
    -------
    df_metasen : pandas.DataFrame
        A pandas DataFrame of the MS pairs with the six columns representing the Metabolite, Sensor, Metabolite.Pathway, Sensor.Pathway, Metabolite.Names, Long.Range.Channel respectively.

    """
    
    data = pkgutil.get_data(__name__, "_data/MetaChatDB/MetaChatDB_"+species+".tsv")
    df_metasen = pd.read_csv(io.BytesIO(data), sep='\t')

    return df_metasen

def scFEA_annotation(
):
    
    data = pkgutil.get_data(__name__, "_data/scFEA/metabo2module.csv")
    met_annota = pd.read_csv(io.BytesIO(data), sep=',')

    return met_annota

def compass_annotation(
):

    data = pkgutil.get_data(__name__, "_data/Compass/met_md.csv")
    met_annota = pd.read_csv(io.BytesIO(data), sep=',')

    return met_annota

def generate_adata_met_scFEA(
    data_path: str
):
    """
    Generate processed metabolite matrix for scFEA analysis.

    Parameters
    ----------
    data_path : str
        Path to the metabolite data file (CSV format).

    Returns
    -------
    adata_met : pandas.DataFrame
        Processed metabolite adata object ready for downstream analysis.
    """
    mat_met = pd.read_csv(data_path, index_col=0)
    met_annota = scFEA_annotation()
    mat_met.columns = met_annota['HMDB.ID']
    mat_met[mat_met < 0] = 0

    adata_met = anndata.AnnData(mat_met)

    return  adata_met

def generate_adata_met_compass(
    compass_output: str,
    score: str = "Balance",
    norm: str = "rank",
    agg: str = "topk_mean",
    topk: int = 5,
    **score_kwargs
):
    """
    Generate processed metabolite matrix for COMPASS analysis using
    reaction-level penalty scores.

    Metabolite-level production / consumption scores are derived from COMPASS
    reaction penalties via stoichiometric aggregation across all internal
    reactions that involve each cytoplasmic metabolite.  Only metabolites
    in the ``[c]`` compartment are retained and mapped to HMDB IDs.

    Reaction metadata (``rxn_md.csv``) and metabolite annotation
    (``met_md.csv``) are loaded from the package's built-in data.

    Parameters
    ----------
    compass_output : str
        Path to the COMPASS output file.
    score : str
        Score column to extract for each metabolite.  One of ``"Prod"``,
        ``"Cons"``, or ``"Balance"`` (default).
        ``"Balance"`` is production minus consumption.
    norm : str
        Per-reaction normalisation across cells before aggregation.
        ``"rank"`` (default) uses percentile rank; ``"minmax"`` uses
        min-max scaling.
    agg : str
        Aggregation method across reactions.  ``"topk_mean"`` (default)
        averages the top-*k* reactions per cell; ``"max"`` takes the
        maximum.
    topk : int
        Number of top reactions used when ``agg="topk_mean"``. Default 5.
    **score_kwargs
        Additional keyword arguments forwarded to
        :func:`_compass_utils.compute_metabolite_scores`, e.g.
        ``exclude_transport=False``.

    Returns
    -------
    adata_met : anndata.AnnData
        AnnData with shape *(cells × metabolites)*.
        ``obs`` index = cell IDs; ``var`` index = HMDB IDs.
    """
    from ._compass_utils import compute_metabolite_scores

    compass_rxn_mat = pd.read_csv(compass_output, sep="\t", index_col=0)

    rxn_md_data = pkgutil.get_data(__name__, "_data/Compass/rxn_md.csv")
    rxn_md_df = pd.read_csv(io.BytesIO(rxn_md_data), index_col=0)

    met_md = compass_annotation()
    met_md_c = met_md[
        (met_md["compartment"] == "[c]") &
        met_md["ID"].str.startswith("HMDB", na=False)
    ].copy()

    cells = compass_rxn_mat.columns
    records = {}  # met (e.g. "h[c]") -> score Series

    for _, row in met_md_c.iterrows():
        query = f"{row['metName']} [c]"
        met_id = row["met"]

        scores_df, _ = compute_metabolite_scores(
            rxn_md_df, compass_rxn_mat, query,
            norm=norm, agg=agg, topk=topk, **score_kwargs
        )

        records[met_id] = scores_df[score]

    met_mat = pd.DataFrame(records, index=cells)
    met_mat[met_mat < 0] = 0

    # Map met IDs to HMDB IDs; drop duplicate HMDB columns (keep first)
    met_to_hmdb = met_md_c.set_index("met")["ID"]
    met_mat.columns = met_mat.columns.map(met_to_hmdb)
    met_mat = met_mat.loc[:, ~met_mat.columns.duplicated(keep="first")]

    return anndata.AnnData(met_mat)

def generate_adata_met_compass_condition(
    compass_outputs: list,
    score: str = "Balance",
    norm: str = "rank",
    agg: str = "topk_mean",
    topk: int = 5,
    **score_kwargs
):
    """
    Generate metabolite matrices for multiple COMPASS conditions with joint ranking.

    All reaction penalty matrices are concatenated before any normalisation
    so that per-reaction rank (or min-max) scaling is computed across all cells
    from all conditions simultaneously.  This makes scores comparable across
    conditions.  Results are then split back and returned as a list of AnnData
    objects in the same order as the input paths.

    Parameters
    ----------
    compass_outputs : list of str
        Paths to the COMPASS reaction penalties TSV files, one per condition.
    score : str
        Score column to extract.  One of ``"Prod"``, ``"Cons"``, or
        ``"Balance"`` (default).
    norm : str
        Per-reaction normalisation across cells.  ``"rank"`` (default)
        or ``"minmax"``.
    agg : str
        Aggregation method.  ``"topk_mean"`` (default) or ``"max"``.
    topk : int
        Number of top reactions used when ``agg="topk_mean"``. Default 5.
    **score_kwargs
        Additional keyword arguments forwarded to
        :func:`_compass_utils.compute_metabolite_scores`.

    Returns
    -------
    list of anndata.AnnData
        One AnnData per condition *(cells × metabolites)*, in the same order
        as ``compass_outputs``.

    Notes
    -----
    Cell IDs do not need to be unique across conditions — duplicate barcodes
    are handled internally with a temporary prefix.  Only reactions present
    in every file (inner join) are used for scoring.
    """
    import warnings
    from ._compass_utils import compute_metabolite_scores

    mats = [pd.read_csv(p, sep="\t", index_col=0) for p in compass_outputs]
    cells_per_cond = [m.columns.tolist() for m in mats]

    # Add a temporary numeric prefix to make column names globally unique,
    # so that identical barcodes across conditions don't collide after concat.
    prefixed_mats = []
    for i, mat in enumerate(mats):
        renamed = mat.copy()
        renamed.columns = [f"_c{i}_" + c for c in mat.columns]
        prefixed_mats.append(renamed)

    # Joint matrix — rank is computed across all cells from all conditions
    n_rxns_before = len(mats[0].index)
    compass_rxn_mat = pd.concat(prefixed_mats, axis=1, join="inner")
    n_dropped = n_rxns_before - len(compass_rxn_mat.index)
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} reaction(s) not shared across all files "
            "were dropped (inner join)."
        )

    rxn_md_data = pkgutil.get_data(__name__, "_data/Compass/rxn_md.csv")
    rxn_md_df = pd.read_csv(io.BytesIO(rxn_md_data), index_col=0)

    met_md = compass_annotation()
    met_md_c = met_md[
        (met_md["compartment"] == "[c]") &
        met_md["ID"].str.startswith("HMDB", na=False)
    ].copy()

    cells = compass_rxn_mat.columns
    records = {}  # met (e.g. "h[c]") -> score Series

    for _, row in met_md_c.iterrows():
        query = f"{row['metName']} [c]"
        met_id = row["met"]

        scores_df, _ = compute_metabolite_scores(
            rxn_md_df, compass_rxn_mat, query,
            norm=norm, agg=agg, topk=topk, **score_kwargs
        )

        records[met_id] = scores_df[score]

    if not records:
        raise ValueError(
            "No cytoplasmic metabolites could be scored. "
            "Verify that the input files and built-in rxn_md are compatible."
        )

    met_mat = pd.DataFrame(records, index=cells)
    met_mat[met_mat < 0] = 0

    # Map met IDs to HMDB IDs; drop duplicate HMDB columns (keep first)
    met_to_hmdb = met_md_c.set_index("met")["ID"]
    met_mat.columns = met_mat.columns.map(met_to_hmdb)
    met_mat = met_mat.loc[:, ~met_mat.columns.duplicated(keep="first")]

    # Split by prefixed column names, then restore original barcodes as index
    result = []
    for orig_cells, renamed_mat in zip(cells_per_cond, prefixed_mats):
        chunk = met_mat.loc[renamed_mat.columns].copy()
        chunk.index = orig_cells
        result.append(anndata.AnnData(chunk))
    return result


def generate_adata_met_mebocost(
    data_path: str
):
    """
    Generate processed metabolite matrix for scFEA analysis.

    Parameters
    ----------
    data_path : str
        Path to the metabolite data file (CSV format).

    Returns
    -------
    met_mat : pandas.DataFrame
        Processed metabolite matrix ready for downstream analysis.
    """

    mat_met = pd.read_csv(data_path, index_col=0)
    mat_met[mat_met < 0] = 0
    adata_met = anndata.AnnData(mat_met.T)

    return adata_met