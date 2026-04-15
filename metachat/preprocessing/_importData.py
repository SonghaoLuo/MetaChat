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
    met_md_c = met_md[met_md["compartment"] == "[c]"].copy()

    cells = compass_rxn_mat.columns
    records = {}  # hmdb_id -> list of score Series

    for _, row in met_md_c.iterrows():
        query = f"{row['metName']} [c]"
        hmdb_id = row["ID"]

        scores_df, details = compute_metabolite_scores(
            rxn_md_df, compass_rxn_mat, query,
            norm=norm, agg=agg, topk=topk, **score_kwargs
        )

        if not details["producing_rxns"] and not details["consuming_rxns"]:
            continue

        records.setdefault(hmdb_id, []).append(scores_df[score])

    # For HMDB IDs with multiple contributing metabolite names, average them
    met_mat = pd.DataFrame(
        {hid: pd.concat(series_list, axis=1).mean(axis=1)
         for hid, series_list in records.items()},
        index=cells,
    )

    return anndata.AnnData(met_mat)

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