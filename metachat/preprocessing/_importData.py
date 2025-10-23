import io
import math
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
    
    data = pkgutil.get_data(__name__, "_data/metabo2module.csv")
    met_annota = pd.read_csv(io.BytesIO(data), sep=',')

    return met_annota

def compass_annotation(
):
    
    data = pkgutil.get_data(__name__, "_data/met_md_new.csv")
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
    data_path: str,
    baseline: Optional[float] = None
):
    """
    Generate processed metabolite matrix for COMPASS analysis.

    Parameters
    ----------
    data_path : str
        Path to the metabolite data file (TSV format).
    baseline : float or None
        Baseline value to normalize the matrix.
        If None, the minimum (rounded down to 1 decimal) is used as baseline.

    Returns
    -------
    adata_met :
        Processed metabolite adata object ready for downstream analysis.
    """
    met_mat_secret = pd.read_csv(data_path, sep="\t", index_col=0)
    met_mat = met_mat_secret
    met_mat = met_mat[met_mat.index.str.endswith("[e]")]
    met_md_new = compass_annotation()
    met_to_id_mapping = dict(zip(met_md_new["met"], met_md_new["ID"]))
    met_mat.index = met_mat.index.map(met_to_id_mapping)
    met_mat = met_mat[~met_mat.index.isna()]
    met_mat = met_mat.groupby(met_mat.index).sum()
    met_mat[met_mat < 0] = 0
    met_mat = -np.log(met_mat+1)
    if baseline is None:
        baseline = met_mat.min().min()
        baseline = math.floor(baseline * 10) / 10
    met_mat = met_mat - baseline 

    adata_met = anndata.AnnData(met_mat.T)

    return  adata_met, baseline

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