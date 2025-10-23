# ============================================================
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional

import anndata
import scanpy as sc
import squidpy as sq
import gseapy as gp

import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from tqdm import tqdm
from multiprocessing import Pool
from scipy import sparse
import matplotlib.pyplot as plt

from ..plotting import plot_communication_flow
from .._utils import leiden_clustering
# ============================================================

# ================ MCC communication summary =================
def summary_communication(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    copy: bool = False
):

    """
    Summarize communication signals by metabolite sets, pathways, or custom lists.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object that has run :func:`mc.tl.metabolic_communication`.
    database_name : str
        Name of the metabolite–sensor interaction database (e.g., "MetaChatDB").
    sum_metabolites : list of str, optional
        List of specific metabolites to summarize communication for.
        Example: ``['HMDB0000148', 'HMDB0000674']``.
    sum_metapathways : list of str, optional
        List of specific metabolic pathways to summarize communication for.
        Example: ``['Alanine, aspartate and glutamate metabolism', 'Glycerolipid Metabolism']``.
    sum_customerlists : dict, optional
        Custom metabolite–sensor groups.  
        Each key represents a custom name, and the value is a list of (metabolite, sensor) tuples.  
        Example: ``{'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')], 
                    'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}``.
    copy : bool, default=False
        Whether to return a modified copy of the :class:`AnnData` object.

    Returns
    -------
    adata : anndata.AnnData
        sum_metabolites, sum_metapathways, sum_customerlists can provided by user in one time.  
        the summary information are added to ``.obsm`` and ``.obsp``. For example:
        For each "metabolite_name" in "sum_metabolites", ``adata.obsp['MetaChat-'+database_name+'-'+metabolite_name]``,``adata.obsm['MetaChat-'+database_name+'-sum-sender-'+'metabolite_name']['s-'+metabolite_name]`` and ``adata.obsm['MetaChat-'+database_name+'-sum-receiver-'+'metabolite_name']['r-'+metabolite_name]``.
        For each "pathway_name" in "sum_metapathways", ``adata.obsp['MetaChat-'+database_name+'-'+pathway_name]``, ``adata.obsm['MetaChat-'+database_name+'-sum-sender-'+'pathway_name']['s-'+pathway_name]`` and ``adata.obsm['MetaChat-'+database_name+'-sum-receiver-'+'pathway_name']['r-'+pathway_name]``.
        For each "customerlist_name" in "sum_customerlists", ``adata.obsp['MetaChat-'+database_name+'-'+customerlist_name]``, ``adata.obsm['MetaChat-'+database_name+'-sum-sender-'+'customerlist_name']['s-'+customerlist_name]`` and ``adata.obsm['MetaChat-'+database_name+'-sum-receiver-'+'customerlist_name']['r-'+customerlist_name]``.
        If copy=True, return the AnnData object and return None otherwise.                          
    """

    # ==== Input checks ====
    assert database_name is not None, "Please specify database_name."
    assert any([sum_metabolites, sum_metapathways, sum_customerlists]), (
        "Please provide at least one of sum_metabolites, sum_metapathways, or sum_customerlists."
    )

    ncell = adata.shape[0]
    df_metasen = adata.uns["df_metasen_filtered"]
    
    # ==== Summarize by metabolites ====
    if sum_metabolites is not None:

        P_sender_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metabolites))]
        P_receiver_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metabolites))]

        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_metabolites))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_metabolites))]

        col_names_sender_all = []
        col_names_receiver_all = []

        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_metabolite in range(len(sum_metabolites)):
            metabolite_name = sum_metabolites[idx_metabolite]
            if metabolite_name in df_metasen['HMDB.ID'].values:
                idx_related = np.where(df_metasen["HMDB.ID"].str.contains(metabolite_name, regex=False, na=False))[0]

                for i in idx_related:
                    P_sender = adata.obsp['MetaChat-' + database_name + '-sender-' + df_metasen.loc[i,'HMDB.ID'] + '-' + df_metasen.loc[i,'Sensor.Gene']]
                    P_receiver = adata.obsp['MetaChat-' + database_name + '-receiver-' + df_metasen.loc[i,'HMDB.ID'] + '-' + df_metasen.loc[i,'Sensor.Gene']]
                    P_sender_list[idx_metabolite] = P_sender_list[idx_metabolite] + P_sender
                    P_receiver_list[idx_metabolite] = P_receiver_list[idx_metabolite] + P_receiver
                    X_sender_list[idx_metabolite] = X_sender_list[idx_metabolite] + np.array(P_sender.sum(axis=1))
                    X_receiver_list[idx_metabolite] = X_receiver_list[idx_metabolite] + np.array(P_receiver.sum(axis=0).T)

                adata.obsp['MetaChat-' + database_name + '-sender-' + metabolite_name] = P_sender_list[idx_metabolite]
                adata.obsp['MetaChat-' + database_name + '-receiver-' + metabolite_name] = P_receiver_list[idx_metabolite]
                X_sender_all = np.concatenate((X_sender_all, X_sender_list[idx_metabolite]), axis=1)
                X_receiver_all = np.concatenate((X_receiver_all, X_receiver_list[idx_metabolite]), axis=1)

                col_names_sender_all.append("s-" + metabolite_name)
                col_names_receiver_all.append("r-" + metabolite_name)
            else:
                print(f"Warning: {metabolite_name} is not in the results")

        df_sender_all = pd.DataFrame(data=X_sender_all, columns=col_names_sender_all, index=adata.obs_names)
        df_receiver_all = pd.DataFrame(data=X_receiver_all, columns=col_names_receiver_all, index=adata.obs_names)

        adata.obsm['MetaChat-' + database_name + '-sum-sender-metabolite'] = df_sender_all
        adata.obsm['MetaChat-' + database_name + '-sum-receiver-metabolite'] = df_receiver_all

    # ==== Summarize by metabolic pathways ====
    if sum_metapathways is not None:

        P_sender_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metapathways))]
        P_receiver_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metapathways))]

        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_metapathways))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_metapathways))]

        col_names_sender_all = []
        col_names_receiver_all = []

        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_pathway in range(len(sum_metapathways)):
            pathway_name = sum_metapathways[idx_pathway]
            if np.sum(df_metasen["Metabolite.Pathway"].str.contains(pathway_name, regex=False, na=False)) > 0:
                idx_related = np.where(df_metasen["Metabolite.Pathway"].str.contains(pathway_name, regex=False, na=False))[0]

                for i in idx_related:
                    P_sender = adata.obsp['MetaChat-' + database_name + '-sender-' + df_metasen.loc[i,'HMDB.ID'] + '-' + df_metasen.loc[i,'Sensor.Gene']]
                    P_receiver = adata.obsp['MetaChat-' + database_name + '-receiver-' + df_metasen.loc[i,'HMDB.ID'] + '-' + df_metasen.loc[i,'Sensor.Gene']]
                    P_sender_list[idx_pathway] = P_sender_list[idx_pathway] + P_sender
                    P_receiver_list[idx_pathway] = P_receiver_list[idx_pathway] + P_receiver
                    X_sender_list[idx_pathway] = X_sender_list[idx_pathway] + np.array(P_sender.sum(axis=1))
                    X_receiver_list[idx_pathway] = X_receiver_list[idx_pathway] + np.array(P_receiver.sum(axis=0).T)
                    
                adata.obsp['MetaChat-' + database_name + '-sender-' + pathway_name] = P_sender_list[idx_pathway]
                adata.obsp['MetaChat-' + database_name + '-receiver-' + pathway_name] = P_receiver_list[idx_pathway]

                X_sender_all = np.concatenate((X_sender_all, X_sender_list[idx_pathway]), axis=1)
                X_receiver_all = np.concatenate((X_receiver_all, X_receiver_list[idx_pathway]), axis=1)

                col_names_sender_all.append("s-" + pathway_name)
                col_names_receiver_all.append("r-" + pathway_name)
            else:
                print(f"Warning: {pathway_name} is not in the results")

        df_sender_all = pd.DataFrame(data=X_sender_all, columns=col_names_sender_all, index=adata.obs_names)
        df_receiver_all = pd.DataFrame(data=X_receiver_all, columns=col_names_receiver_all, index=adata.obs_names)

        adata.obsm['MetaChat-' + database_name + '-sum-sender-pathway'] = df_sender_all
        adata.obsm['MetaChat-' + database_name + '-sum-receiver-pathway'] = df_receiver_all
    
    # ==== Summarize by customer-defined lists ====
    if sum_customerlists is not None:

        P_sender_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_customerlists))]
        P_receiver_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_customerlists))]

        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_customerlists))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_customerlists))]

        col_names_sender_all = []
        col_names_receiver_all = []

        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_customerlist, (customerlist_name, customerlist_value) in enumerate(sum_customerlists.items()):
            for idx_value in customerlist_value:
                temp_meta = idx_value[0]
                temp_sens = idx_value[1]
                P_sender = adata.obsp['MetaChat-' + database_name + '-sender-' + temp_meta + '-' + temp_sens]
                P_receiver = adata.obsp['MetaChat-' + database_name + '-receiver-' + temp_meta + '-' + temp_sens]
                P_sender_list[idx_customerlist] = P_sender_list[idx_customerlist] + P_sender
                P_receiver_list[idx_customerlist] = P_receiver_list[idx_customerlist] + P_receiver
                X_sender_list[idx_customerlist] = X_sender_list[idx_customerlist] + np.array(P_sender.sum(axis=1))
                X_receiver_list[idx_customerlist] = X_receiver_list[idx_customerlist] + np.array(P_receiver.sum(axis=0).T)     

            adata.obsp['MetaChat-' + database_name + '-sender-' + customerlist_name] = P_sender_list[idx_customerlist]
            adata.obsp['MetaChat-' + database_name + '-receiver-' + customerlist_name] = P_receiver_list[idx_customerlist]

            X_sender_all = np.concatenate((X_sender_all, X_sender_list[idx_customerlist]), axis=1)
            X_receiver_all = np.concatenate((X_receiver_all, X_receiver_list[idx_customerlist]), axis=1)
            
            col_names_sender_all.append("s-" + customerlist_name)
            col_names_receiver_all.append("r-" + customerlist_name)

        df_sender_all = pd.DataFrame(data=X_sender_all, columns=col_names_sender_all, index=adata.obs_names)
        df_receiver_all = pd.DataFrame(data=X_receiver_all, columns=col_names_receiver_all, index=adata.obs_names)

        adata.obsm['MetaChat-' + database_name + '-sum-sender-customer'] = df_sender_all
        adata.obsm['MetaChat-' + database_name + '-sum-receiver-customer'] = df_receiver_all

    return adata if copy else None

# ================ MCC flow ================
def communication_flow(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    sum_ms_pairs: list = None,
    spatial_key: str = 'spatial',
    k: int = 5,
    pos_idx: Optional[np.ndarray] = None,
    copy: bool = False
):
    """
    Construct spatial vector fields representing metabolic communication flow.

    Parameters
    ----------
    adata : anndata.AnnData
        The data matrix of shape ``n_obs`` × ``n_var``. If compute MCC flow from specific metabolites, 
        metapathways or customerlists, please run :func:`mc.tl.summary_communication` first.
    database_name : str
        Name of the Metabolite-Sensor interaction database (e.g., ``"MetaChatDB"``).
    sum_metabolites : list of str, optional
        Specific metabolites to summarize and compute flow for.
        For example, sum_metabolites = ['HMDB0000148','HMDB0000674'].
    sum_metapathways : list of str, optional
        Specific metabolic pathways to summarize and compute flow for.
        Example: ``['Alanine, aspartate and glutamate metabolism', 'Glycerolipid Metabolism']``.
    sum_customerlists : dict, optional
        Custom metabolite–sensor pair groups to summarize.
        Example: ``{'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')],
                    'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}``.
    sum_ms_pairs : list of str, optional
        Specific metabolite–sensor pairs, e.g. ``['HMDB0000148-Grm5']``.
    spatial_key : str, default='spatial'
        Key in `.obsm` that contains spatial coordinates.
    k : int, default=5
        Top-k senders/receivers used for computing the flow direction.
    pos_idx : np.ndarray, optional
        Column indices of `.obsm[spatial_key]` to use for flow computation.
        Example: ``np.array([0, 2])`` uses x–z coordinates.
    copy : bool, default=False
        If True, return a modified copy of `adata`; otherwise modify in place.
    
    Returns
    -------
    anndata.AnnData or None
        Adds sender and receiver vector fields to `.obsm`, for example:  
        - ``.obsm['MetaChat-vf-databaseX-sender-HMDB0000148']``  
        - ``.obsm['MetaChat-vf-databaseX-receiver-HMDB0000148']``  
        If ``copy=True``, returns the modified AnnData object; otherwise returns ``None``.
    """
    
    # ---- Input checks ----
    assert database_name is not None, "Please specify database_name."

    obsp_names_sender = []
    obsp_names_receiver = []
    if sum_metabolites is not None:
        for metabolite_name in sum_metabolites:
            obsp_names_sender.append(database_name + '-sender-' + metabolite_name)
            obsp_names_receiver.append(database_name + '-receiver-' + metabolite_name)
    
    if sum_metapathways is not None:
        for pathway_name in sum_metapathways:
            obsp_names_sender.append(database_name + '-sender-' + pathway_name)
            obsp_names_receiver.append(database_name + '-receiver-' + pathway_name)

    if sum_customerlists is not None:
        for customerlist_name in sum_customerlists.keys():
            obsp_names_sender.append(database_name + '-sender-' + customerlist_name)
            obsp_names_receiver.append(database_name + '-receiver-' + customerlist_name)
    
    if sum_ms_pairs is not None:
        for ms_pair in sum_ms_pairs:
            obsp_names_sender.append(database_name + '-sender-' + ms_pair)
            obsp_names_receiver.append(database_name + '-receiver-' + ms_pair)

    obsp_names_sender.append(f"{database_name}-sender-total-total")
    obsp_names_receiver.append(f"{database_name}-receiver-total-total")

    if all(x is None for x in [sum_metabolites, sum_metapathways, sum_customerlists, sum_ms_pairs]):
        print("No subset specified — computing MCC for all signals.")

    pts = np.array( adata.obsm[spatial_key], float)
    if pos_idx is not None:
        pts = pts[:,pos_idx]

    for i in range(len(obsp_names_sender)):
        key_sender = 'MetaChat-'+obsp_names_sender[i]
        key_receiver = 'MetaChat-'+obsp_names_receiver[i]

        if not key_sender in adata.obsp.keys():
            raise KeyError(f"Please check whether the mc.tl.summary_communication function run or whether {key_sender} are in adata.obsp.keys().")
        P_sender = adata.obsp[key_sender]
        P_receiver = adata.obsp[key_receiver]
        P_sum_sender = np.array(P_sender.sum(axis=1)).reshape(-1)
        P_sum_receiver = np.array(P_receiver.sum(axis=0)).reshape(-1)

        sender_vf = np.zeros_like(pts)
        receiver_vf = np.zeros_like(pts)

        S_lil = P_sender.tolil()
        for j in range(P_sender.shape[0]):
            if len(S_lil.rows[j]) <= k:
                tmp_idx = np.array( S_lil.rows[j], int )
                tmp_data = np.array( S_lil.data[j], float )
            else:
                row_np = np.array( S_lil.rows[j], int )
                data_np = np.array( S_lil.data[j], float )
                sorted_idx = np.argsort( -data_np )[:k]
                tmp_idx = row_np[ sorted_idx ]
                tmp_data = data_np[ sorted_idx ]
            if len(tmp_idx) == 0:
                continue
            elif len(tmp_idx) == 1:
                avg_v = pts[tmp_idx[0],:] - pts[j,:]
            else:
                tmp_v = pts[tmp_idx,:] - pts[j,:]
                tmp_v = normalize(tmp_v, norm='l2')
                avg_v = tmp_v * tmp_data.reshape(-1,1)
                avg_v = np.sum( avg_v, axis=0 )
            avg_v = normalize( avg_v.reshape(1,-1) )
            sender_vf[j,:] = avg_v[0,:] * P_sum_sender[j]
        
        S_lil = P_receiver.T.tolil()
        for j in range(P_receiver.shape[1]):
            if len(S_lil.rows[j]) <= k:
                tmp_idx = np.array( S_lil.rows[j], int )
                tmp_data = np.array( S_lil.data[j], float )
            else:
                row_np = np.array( S_lil.rows[j], int )
                data_np = np.array( S_lil.data[j], float )
                sorted_idx = np.argsort( -data_np )[:k]
                tmp_idx = row_np[ sorted_idx ]
                tmp_data = data_np[ sorted_idx ]
            if len(tmp_idx) == 0:
                continue
            elif len(tmp_idx) == 1:
                avg_v = -pts[tmp_idx,:] + pts[j,:]
            else:
                tmp_v = -pts[tmp_idx,:] + pts[j,:]
                tmp_v = normalize(tmp_v, norm='l2')
                avg_v = tmp_v * tmp_data.reshape(-1,1)
                avg_v = np.sum( avg_v, axis=0 )
            avg_v = normalize( avg_v.reshape(1,-1) )
            receiver_vf[j,:] = avg_v[0,:] * P_sum_receiver[j]

        adata.obsm["MetaChat-vf-"+obsp_names_sender[i]] = sender_vf
        adata.obsm["MetaChat-vf-"+obsp_names_receiver[i]] = receiver_vf

    return adata if copy else None


# ================ Group-level MCC ================
def _summarize_group(X, clusterid, clusternames, n_permutations=100):
    # Input a sparse matrix of cell signaling and output a pandas dataframe
    # for group-group signaling
    n = len(clusternames)
    X_cluster = np.empty([n,n], float)
    p_cluster = np.zeros([n,n], float)
    for i in range(n):
        tmp_idx_i = np.where(clusterid==clusternames[i])[0]
        for j in range(n):
            tmp_idx_j = np.where(clusterid==clusternames[j])[0]
            X_cluster[i,j] = X[tmp_idx_i,:][:,tmp_idx_j].mean()
    for i in range(n_permutations):
        clusterid_perm = np.random.permutation(clusterid)
        X_cluster_perm = np.empty([n,n], float)
        for j in range(n):
            tmp_idx_j = np.where(clusterid_perm==clusternames[j])[0]
            for k in range(n):
                tmp_idx_k = np.where(clusterid_perm==clusternames[k])[0]
                X_cluster_perm[j,k] = X[tmp_idx_j,:][:,tmp_idx_k].mean()
        p_cluster[X_cluster_perm >= X_cluster] += 1.0
    p_cluster = p_cluster / n_permutations
    df_cluster = pd.DataFrame(data=X_cluster, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster, index=clusternames, columns=clusternames)
    return df_cluster, df_p_value

def _init_communication_group(_adata):
    global adata
    adata = _adata

def _compute_group_result(args):
    group_name, clusterid, celltypes, summary, obsp_name, n_permutations = args
    key = 'MetaChat-' + obsp_name
    S = adata.obsp[key]
    tmp_df, tmp_p_value = _summarize_group(S, clusterid, celltypes, n_permutations)
    uns_key = 'MetaChat_group-' + group_name + '-' + obsp_name
    return (uns_key, {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value})

def communication_group(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    summary: str = 'sender',
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    sum_ms_pairs: list = None,
    n_permutations: int = 100,
    use_parallel: bool = True,
    n_jobs: int = 16,
    copy: bool = False
):
    """
    Summarize metabolic communication to group-level MCC and compute p-values via label permutation.

    Parameters
    ----------
    adata : anndata.AnnData
        The data matrix of shape ``n_obs`` × ``n_var``. If compute MCC flow from specific metabolites, 
        metapathways or customerlists, please run :func:`mc.tl.summary_communication` first.
    database_name : str
        Name of the Metabolite-Sensor interaction database.
    group_name : str
        Column key in ``adata.obs`` specifying cell or spot group labels.
    summary : {'sender', 'receiver'}, default='sender'
        Whether to summarize sender- or receiver-side communication.
    sum_metabolites : list of str, optional
        List of specific metabolites to summarize, e.g. ``['HMDB0000148', 'HMDB0000674']``.
    sum_metapathways : list of str, optional
        List of metabolic pathways to summarize, e.g. ``['Alanine, aspartate and glutamate metabolism', 'Glycerolipid Metabolism']``.
    sum_customerlists : dict, optional
        Custom metabolite–sensor pair groups to summarize.
        Example: ``{'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')],
                    'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}``.
    sum_ms_pairs : list of str, optional
        Specific metabolite–sensor pairs, e.g. ``['HMDB0000148-Grm5']``.
    n_permutations : int, default=100
        Number of random label permutations for p-value estimation.
    use_parallel : bool, default=True
        Whether to use multiprocessing.
    n_jobs : int, default=16
        Number of parallel worker processes.
    copy : bool, default=False
        If True, return a copy of the AnnData object; otherwise modify in place.
    
    Returns
    -------
    anndata.AnnData or None
        Adds group-level communication results into:
        ``.uns['MetaChat_group-{group_name}-{database_name}-{item_name}']``  
        Each key contains a dict with:
        - ``['communication_matrix']`` : group × group MCC intensity  
        - ``['communication_pvalue']`` : permutation-based p-values  
        If ``copy=True``, returns the modified AnnData object.
    """

    # ==== Input checks ====
    assert database_name is not None, "Please specify database_name."
    assert group_name is not None, "Please specify group_name."

    celltypes = list(adata.obs[group_name].unique())
    celltypes.sort()
    for i in range(len(celltypes)):
        celltypes[i] = str(celltypes[i])
    clusterid = np.array(adata.obs[group_name], str)

    obsp_names = []
    if sum_metabolites is not None:
        for metabolite_name in sum_metabolites:
            obsp_names.append(database_name + '-' + summary + '-' + metabolite_name)
    
    if sum_metapathways is not None:
        for pathway_name in sum_metapathways:
            obsp_names.append(database_name + '-' + summary + '-' + pathway_name)

    if sum_customerlists is not None:
        for customerlist_name in sum_customerlists.keys():
            obsp_names.append(database_name + '-' + summary + '-' + customerlist_name)

    if sum_ms_pairs is not None:
        for ms_pairs_name in sum_ms_pairs:
            obsp_names.append(database_name + '-' + summary + '-' + ms_pairs_name)        

    obsp_names.append(database_name + '-' + summary + '-total-total')
    
    if all(x is None for x in [sum_metabolites, sum_metapathways, sum_customerlists, sum_ms_pairs]):
        print("No specific summary provided — computing group-level MCC for all signals.")
    
    # Check keys
    for i in range(len(obsp_names)):
        key = 'MetaChat-'+obsp_names[i]
        if not key in adata.obsp.keys():
            raise KeyError(f"Please check whether the mc.tl.summary_communication function run or whether {key} are in adata.obsp.keys().")

    task_list = [(group_name, clusterid, celltypes, summary, name, n_permutations) for name in obsp_names]
    results = []

    if use_parallel:
        with Pool(processes=n_jobs, initializer=_init_communication_group, initargs=(adata,)) as pool:
            with tqdm(total=len(task_list), desc="  Computing group-level MCC", dynamic_ncols=True) as pbar:
                for result in pool.imap_unordered(_compute_group_result, task_list):
                    results.append(result)
                    pbar.update(1)
    else:
        for key in tqdm(obsp_names, desc="  Computing group-level MCC", dynamic_ncols=True):
            results.append(_compute_group_result(key))

    # Save results into adata.uns
    for uns_key, result_dict in results:
        adata.uns[uns_key] = result_dict
    
    return adata if copy else None

def _init_spatial_permutation(_S_list, _bin_positions, _index_obsp_list, _bin_counts_ij, _bin_total_counts_ij):

    global S_list
    global bin_positions
    global index_obsp_list
    global bin_counts_ij
    global bin_total_counts_ij

    S_list = _S_list
    bin_positions = _bin_positions
    index_obsp_list = _index_obsp_list
    bin_counts_ij = _bin_counts_ij
    bin_total_counts_ij = _bin_total_counts_ij

def _compute_spatial_group_result(args):

    i, j, _ = args  # trial_idx is not used since each call is independent

    result = {}
    for idx in index_obsp_list:
        S = S_list[idx]
        all_sampled_pos = []
        for bin_id, count in bin_counts_ij[i][j].items():
            positions = bin_positions[bin_id]
            if len(positions) < count:
                continue
            sampled_idx = np.random.choice(len(positions), count, replace=False)
            all_sampled_pos.append(positions[sampled_idx])

        if len(all_sampled_pos) == 0:
            result[idx] = 0.0
            continue

        all_sampled_pos = np.concatenate(all_sampled_pos, axis=0)
        row_idx, col_idx = all_sampled_pos[:, 0], all_sampled_pos[:, 1]
        result[idx] = S[row_idx, col_idx].sum() / bin_total_counts_ij[i][j] if bin_total_counts_ij[i][j] > 0 else 0

    return (i, j, result)

def communication_group_spatial(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    summary: str = 'sender',
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    sum_ms_pairs: list = None,
    n_permutations: int = 100,
    bins_num: int = 30,
    use_parallel: bool = True,
    n_jobs: int = 16,
    copy: bool = False
):    
    """
    Function for summarizing metabolic MCC communication to group-level communication
    and computing p-values based on spatial distance distribution.

    Parameters
    ----------
    adata : anndata.AnnData
        The data matrix with shape ``n_obs`` × ``n_var``. 
        If compute MCC flow from specific metabolites, metapathways or customerlists, 
        please run :func:`mc.tl.summary_communication` first.
    database_name : str
        Name of the Metabolite-Sensor interaction database.
    group_name : str
        Key of cell/spot group annotation in ``adata.obs``.
    summary : str, optional
        'sender' or 'receiver'; defines which communication direction to summarize.
    sum_metabolites : list, optional
        List of specific metabolites to summarize communication for. 
        Example: ['HMDB0000148','HMDB0000674'].
    sum_metapathways : list, optional
        List of specific metabolic pathways to summarize communication for.
        Example: ['Alanine, aspartate and glutamate metabolism','Glycerolipid Metabolism'].
    sum_customerlists : dict, optional
        Custom dictionaries of metabolite–sensor pairs.
        Example: {'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')],
                  'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}
    sum_ms_pairs : list of str, optional
        Specific metabolite–sensor pairs, e.g. ``['HMDB0000148-Grm5']``.
    n_permutations : int, optional
        Number of label permutations for computing p-values (default: 100).
    bins_num : int, optional
        Number of bins for sampling based on spatial distance distribution (default: 30).
    use_parallel : bool, optional
        Whether to run the computation in parallel using multiprocessing (default: True).
    n_jobs : int, optional
        Number of worker processes for parallelization (default: 16).
    copy : bool, optional
        Whether to return a modified copy of the :class:`anndata.AnnData` object.
    
    Returns
    -------
    anndata.AnnData or None
        Adds group-level communication results to:
        ``.uns['MetaChat_group_spatial-{group_name}-{database_name}-{signal_name}']``,
        where each entry is a dictionary containing:
            - ``'communication_matrix'`` : mean MCC intensity between groups
            - ``'communication_pvalue'`` : permutation-based p-value matrix
        Returns the AnnData object if ``copy=True``, otherwise modifies in place.
    """

    # ==== Check inputs ====
    assert database_name is not None, "Please specify database_name."
    assert group_name is not None, "Please specify group_name."

    celltypes = sorted(map(str, adata.obs[group_name].unique()))
    clusterid = np.array(adata.obs[group_name], str)

    obsp_names = []
    if sum_metabolites is not None:
        for metabolite_name in sum_metabolites:
            obsp_names.append(database_name + '-' + summary + '-' + metabolite_name)

    if sum_metapathways is not None:
        for pathway_name in sum_metapathways:
            obsp_names.append(database_name + '-' + summary + '-' + pathway_name)

    if sum_customerlists is not None:
        for customerlist_name in sum_customerlists.keys():
            obsp_names.append(database_name + '-' + summary + '-' + customerlist_name)
    
    if sum_ms_pairs is not None:
        for ms_pairs_name in sum_ms_pairs:
            obsp_names.append(database_name + '-' + summary + '-' + ms_pairs_name)     

    obsp_names.append(database_name + '-' + summary + '-total-total')

    if all(x is None for x in [sum_metabolites, sum_metapathways, sum_customerlists, sum_ms_pairs]):
        print("No specific summary provided — computing group-level MCC for all signals.")

    # Check keys
    for i in range(len(obsp_names)):
        key = 'MetaChat-'+obsp_names[i]
        if not key in adata.obsp.keys():
            raise KeyError(f"Please check whether the mc.tl.summary_communication function run or whether {key} are in adata.obsp.keys().")

    dist_matrix = adata.obsp['spatial_distance_LRC_base']
    hist, bin_edges = np.histogram(dist_matrix, bins=bins_num)
    dist_matrix_bin = np.digitize(dist_matrix, bin_edges) - 1
    bin_positions = {category: np.argwhere(dist_matrix_bin == category) for category in range(bins_num + 1)}

    n = len(celltypes)
    bin_counts_ij = [[{} for _ in range(n)] for _ in range(n)]
    bin_total_counts_ij = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            tmp_i = np.where(clusterid == celltypes[i])[0]
            tmp_j = np.where(clusterid == celltypes[j])[0]
            tmp_bin = dist_matrix_bin[tmp_i,:][:,tmp_j].flatten()
            bin_counts_ij[i][j] = Counter(tmp_bin)
            bin_total_counts_ij[i,j] = len(tmp_i) * len(tmp_j)

    S = {}
    X_cluster = {}
    p_cluster = {}

    for idx, name in enumerate(obsp_names):
        key = 'MetaChat-' + name
        S[idx] = adata.obsp[key]
        tmp_matrix = np.zeros((n, n))
        for i in range(n):
            tmp_i = np.where(clusterid == celltypes[i])[0]
            for j in range(n):
                tmp_j = np.where(clusterid == celltypes[j])[0]
                tmp_matrix[i, j] = S[idx][tmp_i][:, tmp_j].mean()
        X_cluster[idx] = tmp_matrix
        p_cluster[idx] = np.zeros((n, n))
    
    S_list = [S[idx] for idx in range(len(obsp_names))]
    index_obsp_list = list(range(len(obsp_names)))

    perm_tasks = []
    for i, j in itertools.product(range(n), repeat=2):
        perm_tasks.extend([(i, j, trial_idx) for trial_idx in range(n_permutations)])

    # Initialize global variables once for the pool
    results = []
    if use_parallel:
        with Pool(processes=n_jobs, initializer=_init_spatial_permutation,
                initargs=(S_list, bin_positions, index_obsp_list, bin_counts_ij, bin_total_counts_ij)) as pool:
            with tqdm(total=len(perm_tasks), desc="  Computing group-level MCC", dynamic_ncols=True) as pbar:
                for result in pool.imap_unordered(_compute_spatial_group_result, perm_tasks):
                    results.append(result)
                    pbar.update(1)
    else:
        results = [_compute_spatial_group_result(task) for task in perm_tasks]

    # Aggregate results into null distributions
    null_dict = {(i, j): {idx: [] for idx in index_obsp_list} for i in range(n) for j in range(n)}
    for i, j, res in results:
        for idx in res:
            null_dict[(i, j)][idx].append(res[idx])

    # Compute p-values
    for i in range(n):
        for j in range(n):
            for idx in index_obsp_list:
                null_dist = np.array(null_dict[(i, j)][idx])
                p_val = np.sum(null_dist >= X_cluster[idx][i, j]) / n_permutations
                p_cluster[idx][i, j] = p_val

    for idx, name in enumerate(obsp_names):
        df_cluster = pd.DataFrame(X_cluster[idx], index=celltypes, columns=celltypes)
        df_pval = pd.DataFrame(p_cluster[idx], index=celltypes, columns=celltypes)
        adata.uns[f'MetaChat_group_spatial-{group_name}-{name}'] = {
            'communication_matrix': df_cluster,
            'communication_pvalue': df_pval
        }
    
    return adata if copy else None

# ================ MCC pathway summary ================
def summary_pathway(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    summary: str = 'sender',
    sender_group: str = None,
    receiver_group: str = None,
    permutation_spatial: bool = False
):
    """
    Summarize MCC (Metabolite–Sensor Communication) patterns between specific sender
    and receiver groups, and rank metabolic and sensor pathways.

    Parameters
    ----------
    adata : anndata.AnnData
        The data matrix of shape ``n_obs`` × ``n_var``.
    database_name : str
        Name of the Metabolite-Sensor interaction database.
    group_name : str
        Group name of the cell annotation previously saved in ``adata.obs``. 
    summary : str, default='sender'
        The communication summary type ('sender' or 'receiver').
    sender_group : str
        Name of the sender group
    receiver_group : str
        Name of the receiver group
    permutation_spatial : bool, default=False
        Whether to use results from ``mc.tl.communication_group_spatial``.
    
    Returns
    -------
    metapathway_rank : pd.DataFrame
        Ranking of metabolic pathways by communication score and p-value.
    senspathway_rank : pd.DataFrame
        Ranked sensor pathways by HITS authority score.
    ms_result : pd.DataFrame
        Matrix of summed communication intensity between metabolic and sensor pathways.
    metapathway_pair_contributions : dict[str, pd.DataFrame]
        For each metabolic pathway, detailed metabolite–sensor pair contributions.
    """
    
    # ==== Check inputs ====
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."
    assert sender_group is not None, "Please at least specify sender_group."
    assert receiver_group is not None, "Please at least specify receiver_group."

    df_metasen = adata.uns["df_metasen_filtered"].copy()
    Metapathway_data = df_metasen["Metabolite.Pathway"].copy()
    Metapathway_list = []
    for item in Metapathway_data:
        split_items = item.split('; ')
        Metapathway_list.extend(split_items)
    sum_metapathway = np.unique(Metapathway_list).tolist()
    sum_metapathway = [x for x in sum_metapathway if x != 'nan']

    # Choose the most significant metabolic pathway in the communication between these sender group and receiver group
    MCC_metapathway = pd.DataFrame(np.zeros((len(sum_metapathway),2)), index=sum_metapathway, columns=['communication_score','p_value'])
    for pathway_name in MCC_metapathway.index:
        if permutation_spatial == True:
            key = "MetaChat_group_spatial-" + group_name + "-" + database_name + "-" + summary + "-" + pathway_name
            if not key in adata.uns.keys():
                raise KeyError(f"Please check whether the mc.tl.communication_group_spatial function are run and whether {key} are in adata.uns.keys()." \
                               "Note that this function needs to compute the group-level for all pathways")
            MCC_metapathway.loc[pathway_name,"communication_score"] = adata.uns[key]["communication_matrix"].loc[sender_group,receiver_group]
            MCC_metapathway.loc[pathway_name,"p_value"] = adata.uns[key]["communication_pvalue"].loc[sender_group,receiver_group]
        else:
            key = "MetaChat_group-" + group_name + "-" + database_name + "-" + summary + "-" + pathway_name
            if not key in adata.uns.keys():
                raise KeyError(f"Please check whether the mc.tl.communication_group function are run and whether {key} are in adata.uns.keys()." \
                               "Note that this function needs to compute the group-level for all pathways")
            MCC_metapathway.loc[pathway_name,"communication_score"] = adata.uns[key]["communication_matrix"].loc[sender_group,receiver_group]
            MCC_metapathway.loc[pathway_name,"p_value"] = adata.uns[key]["communication_pvalue"].loc[sender_group,receiver_group]
      
    metapathway_rank = MCC_metapathway.sort_values(by=['p_value', 'communication_score'], ascending=[True, False])
    metapathway_rank = metapathway_rank.reset_index().rename(columns={'index': 'Metabolite.Pathway'})

    # Compute the each m-s pairs communication_score
    MCC_group_pair = adata.uns['df_metasen_filtered'].copy()
    for irow, ele in MCC_group_pair.iterrows():
        Metaname = ele['HMDB.ID']
        Sensname = ele['Sensor.Gene']
        key = "MetaChat_group-" + group_name + "-" + database_name + "-" + summary + "-" + Metaname + "-" + Sensname
        if not key in adata.uns.keys():
                raise KeyError(f"Please check whether the mc.tl.communication_group function are run and whether {key} are in adata.uns.keys()." \
                               "Note that this function needs to compute the group-level for all m-s pairs")
        MCC_group_pair.loc[irow, "communication_score"] = adata.uns[key]["communication_matrix"].loc[sender_group,receiver_group]

    MCC_Meta2pathway = MCC_group_pair[["HMDB.ID", "Metabolite.Pathway", "Sensor.Gene", "Sensor.Pathway", "communication_score"]]
    MCC_Meta2pathway = MCC_Meta2pathway[MCC_Meta2pathway['Metabolite.Pathway'].notna() & MCC_Meta2pathway['Sensor.Pathway'].notna()]
    MCC_Meta2pathway['Metabolite.Pathway'] = MCC_Meta2pathway['Metabolite.Pathway'].str.split('; ')
    MCC_Meta2pathway_expanded1 = MCC_Meta2pathway.explode('Metabolite.Pathway')
    MCC_Meta2pathway_expanded1['Sensor.Pathway'] = MCC_Meta2pathway_expanded1['Sensor.Pathway'].str.split('; ')
    MCC_Meta2pathway_expanded2 = MCC_Meta2pathway_expanded1.explode('Sensor.Pathway')
    MCC_Meta2pathway_group = MCC_Meta2pathway_expanded2.groupby(['Metabolite.Pathway', 'Sensor.Pathway'], as_index=False).agg({'communication_score': 'sum'})
    
    # Initialize the dictionary to store contributions
    metapathway_pair_contributions = {}

    # Filter the necessary columns from the original df
    pair_info_cols = ["HMDB.ID", "Metabolite.Name", "Sensor.Gene", "Metabolite.Pathway", "Sensor.Pathway", "communication_score"]
    MCC_Meta2pathway_pairs = MCC_group_pair[pair_info_cols].copy()

    # Clean & explode pathways
    MCC_Meta2pathway_pairs = MCC_Meta2pathway_pairs[
        MCC_Meta2pathway_pairs['Metabolite.Pathway'].notna() & MCC_Meta2pathway_pairs['Sensor.Pathway'].notna()
    ].copy()

    MCC_Meta2pathway_pairs['Metabolite.Pathway'] = MCC_Meta2pathway_pairs['Metabolite.Pathway'].str.split('; ')
    MCC_Meta2pathway_pairs['Sensor.Pathway'] = MCC_Meta2pathway_pairs['Sensor.Pathway'].str.split('; ')
    MCC_Meta2pathway_pairs = MCC_Meta2pathway_pairs.explode('Metabolite.Pathway')
    MCC_Meta2pathway_pairs = MCC_Meta2pathway_pairs.explode('Sensor.Pathway')

    # Iterate and store contributions per pathway
    for pathname in MCC_Meta2pathway_group['Metabolite.Pathway'].unique():
        pathway_df = MCC_Meta2pathway_pairs[MCC_Meta2pathway_pairs['Metabolite.Pathway'] == pathname].copy()
        if not pathway_df.empty:
            metapathway_pair_contributions[pathname] = pathway_df[[
                'HMDB.ID', 'Metabolite.Name', 'Sensor.Gene', 'communication_score'
            ]].drop_duplicates().sort_values(by='communication_score', ascending=False).reset_index(drop=True)

    # construct graph network to measure importance
    G = nx.DiGraph()
    edges_with_weights = [
        (row['Metabolite.Pathway'], row['Sensor.Pathway'], row['communication_score']) 
        for _, row in MCC_Meta2pathway_group.iterrows()
    ]
    for edge in edges_with_weights:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    hubs, authorities = nx.hits(G, max_iter=500, normalized=True)
    senspathway_rank = sorted(authorities.items(), key=lambda item: item[1], reverse=True)
    senspathway_rank = pd.DataFrame(senspathway_rank, columns=['Sensor.Pathway', 'Rankscore'])
    senspathway_rank = senspathway_rank[senspathway_rank['Sensor.Pathway'].str.startswith('WP')]
    senspathway_rank = senspathway_rank.reset_index().drop(columns=['index'])

    ms_result = MCC_Meta2pathway_group.pivot_table(index='Metabolite.Pathway', columns='Sensor.Pathway', values='communication_score')
    ms_result = ms_result.fillna(0)

    return metapathway_rank, senspathway_rank, ms_result, metapathway_pair_contributions

# ================= MCC remodelling =================
def communication_responseGenes(
    adata: anndata.AnnData,
    adata_raw: anndata.AnnData,
    database_name: str = None,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    ms_pairs_name: str = None,
    group_name: str = None,
    subgroup: list = None,
    summary: str = 'receiver',
    var_genes = None,
    n_deg_genes: int = None,
    nknots: int = 6,
    n_points: int = 50,
    deg_pvalue_cutoff: float = 0.05,
):
    """
    Identify signal-dependent genes responding to MCC communication patterns.

    Parameters
    ----------
    adata : anndata.AnnData
        adata.AnnData object after running inference function ``mc.tl.metabolic_communication``.
    adata_raw : anndata.AnnData
        adata.AnnData object with raw spatial transcriptome data.
    database_name : str
        Name of the Metabolite-Sensor interaction database.
    metabolite_name : str, optional
        Name of a specific metabolite to detect response genes. For example, metabolite_name = 'HMDB0000148'.
    metapathway_name : str, optional
        Name of a specific metabolic pathways to detect response genes. For example, metabolite_name = 'Alanine, aspartate and glutamate metabolism'.
    customerlist_name : str, optional
        Name of a specific customerlist to detect response genes. For example, customerlist_name = 'CustomerA'.
    ms_pairs_name : str, optional
        Name of a specific metabolite-sensor pairs to detect response genes. For example, ms_pairs_name = 'HMDB0000148-Grm5'.
    group_name : str, optional
        Grouping column name in ``adata.obs`` for selecting subgroups.
    subgroup : list, optional
        Subset of groups to include.
    summary : {'sender', 'receiver'}, default='receiver'
        Specify whether to analyze sender or receiver side.
    n_var_genes
        The number of most variable genes to test.
    var_genes
        The genes to test. n_var_genes will be ignored if given.
    n_deg_genes
        The number of top deg genes to evaluate yhat.
    nknots
        Number of knots in spline when constructing GAM.
    n_points
        Number of points on which to evaluate the fitted GAM 
        for downstream clustering and visualization.
    deg_pvalue_cutoff
        The p-value cutoff of genes for obtaining the fitted gene expression patterns.

    Returns
    -------
    df_deg: pd.DataFrame
        A data frame of deg analysis results, including Wald statistics, degree of freedom, and p-value.
    df_yhat: pd.DataFrame
        A data frame of smoothed gene expression values.   
    """

    # setup R environment
    import rpy2
    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    import rpy2.rinterface_lib.callbacks
    import logging
    rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
    
    ro.r('library(tradeSeq)')
    ro.r('library(clusterExperiment)')
    anndata2ri.activate()
    ro.numpy2ri.activate()
    ro.pandas2ri.activate()
    
    adata_deg_raw = adata_raw.copy()
    adata_deg_var = adata_raw.copy()

    sc.pp.filter_genes(adata_deg_var, min_cells=3)
    sc.pp.filter_genes(adata_deg_raw, min_cells=3)
    sc.pp.normalize_total(adata_deg_var, target_sum=1e5)
    sc.pp.log1p(adata_deg_var)

    sq.gr.spatial_neighbors(adata_deg_var)
    sq.gr.spatial_autocorr(
        adata_deg_var,
        mode="moran",
        n_perms=100,
        n_jobs=1,
    )

    moranI = adata_deg_var.uns['moranI']
    moranI_filtered = moranI[moranI['pval_norm']< 0.05]
    genes = moranI_filtered.index

    if var_genes is None:
        adata_deg_raw = adata_deg_raw[:, genes]
    else:
        adata_deg_raw = adata_deg_raw[:, var_genes]
    del adata_deg_var

    adata_processed = adata.copy()
    if subgroup is not None and group_name is not None:
        adata_processed = adata_processed[adata_processed.obs[group_name].isin(subgroup)].copy()
    adata_deg_raw = adata_deg_raw[adata_processed.obs_names].copy()

    if summary == 'sender':
        summary_abbr = 's'
    else:
        summary_abbr = 'r'

    non_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name, ms_pairs_name])
    if non_none_count > 1:
        raise ValueError("Only one of 'metabolite_name', 'metapathway_name', or 'customerlist_name' can be specified.")

    if metabolite_name is None and metapathway_name is None and customerlist_name is None and ms_pairs_name is None:
        sum_name = 'total-total'
        obsm_name = ''
    elif metabolite_name is not None:
        sum_name = metabolite_name
        obsm_name = '-metabolite'
    elif metapathway_name is not None:
        sum_name = metapathway_name
        obsm_name = '-pathway'
    elif customerlist_name is not None:
        sum_name = customerlist_name
        obsm_name = '-customer'
    elif ms_pairs_name is not None:
        sum_name = ms_pairs_name
        obsm_name = ''

    comm_sum = adata_processed.obsm['MetaChat-' + database_name + "-sum-" + summary + obsm_name][summary_abbr + '-' + sum_name].values.reshape(-1,1)
    cell_weight = np.ones_like(comm_sum).reshape(-1,1)

    # send adata to R
    adata_r = anndata2ri.py2rpy(adata_deg_raw)
    ro.r.assign("adata", adata_r)
    ro.r("X <- as.matrix( assay( adata, 'X') )")
    ro.r.assign("pseudoTime", comm_sum)
    ro.r.assign("cellWeight", cell_weight)

    # perform analysis (tradeSeq-1.0.1 in R-3.6.3)
    string_fitGAM = 'sce <- fitGAM(counts=X, pseudotime=pseudoTime[,1], cellWeights=cellWeight[,1], nknots=%d, verbose=TRUE)' % nknots
    ro.r(string_fitGAM)
    ro.r('assoRes <- data.frame( associationTest(sce, global=FALSE, lineage=TRUE) )')
    ro.r('assoRes <- assoRes[!is.na(assoRes[,"waldStat_1"]),]')

    with localconverter(ro.pandas2ri.converter):
        df_assoRes = ro.r['assoRes']
    ro.r('assoRes = assoRes[assoRes[,"pvalue_1"] <= %f,]' % deg_pvalue_cutoff)
    ro.r('oAsso <- order(assoRes[,"waldStat_1"], decreasing=TRUE)')
    if n_deg_genes is None:
        n_deg_genes = df_assoRes.shape[0]
    string_cluster = 'clusPat <- clusterExpressionPatterns(sce, nPoints = %d,' % n_points\
        + 'verbose=TRUE, genes = rownames(assoRes)[oAsso][1:min(%d,length(oAsso))],' % n_deg_genes \
        + ' k0s=4:5, alphas=c(0.1))'
    ro.r(string_cluster)
    ro.r('yhatScaled <- data.frame(clusPat$yhatScaled)')
    with localconverter(ro.pandas2ri.converter):
        yhat_scaled = ro.r['yhatScaled']

    df_deg = df_assoRes.rename(columns={'waldStat_1':'waldStat', 'df_1':'df', 'pvalue_1':'pvalue'})
    idx = np.argsort(-df_deg['waldStat'].values)
    df_deg = df_deg.iloc[idx]
    df_yhat = yhat_scaled

    anndata2ri.deactivate()
    ro.numpy2ri.deactivate()
    ro.pandas2ri.deactivate()

    return df_deg, df_yhat
    
def communication_responseGenes_cluster(
    df_deg: pd.DataFrame,
    df_yhat: pd.DataFrame,
    deg_clustering_npc: int = 10,
    deg_clustering_knn: int = 5,
    deg_clustering_res: float = 1.0,
    n_deg_genes: int = 200,
    p_value_cutoff: float = 0.05
):
    """
    Function for cluster the communcation DE genes based on their fitted expression pattern.

    Parameters
    ----------
    df_deg : pd.DataFrame
        DEG summary DataFrame from ``mc.tl.communication_responseGenes``.
        Each row corresponds to a tested gene with columns including:
        "waldStat" (Wald statistics), "df" (degrees of freedom), and "pvalue" (p-value).
    df_yhat : pd.DataFrame
        Smoothed gene expression patterns (fitted values) from ``mc.tl.communication_responseGenes``.
    deg_clustering_npc : int, default=10
        Number of principal components to retain for clustering.
    deg_clustering_knn : int, default=5
        Number of neighbors when constructing the KNN graph for Leiden clustering.
    deg_clustering_res : float, default=1.0
        Resolution parameter for Leiden clustering.
    n_deg_genes : int, default=200
        Number of top DE genes (ranked by Wald statistics) to include in clustering.
    p_value_cutoff : float, default=0.05
        p-value cutoff for selecting DE genes to cluster.

    Returns
    -------
    df_deg_clus: pd.DataFrame
        Metadata table of clustered DE genes, including columns:
        ['waldStat', 'df', 'pvalue', 'cluster'].
    df_yhat_clus: pd.DataFrame
        The fitted gene expression patterns of the clustered genes
    """

    df_deg = df_deg[df_deg['pvalue'] <= p_value_cutoff].copy()
    n_deg_genes = min(n_deg_genes, df_deg.shape[0])
    idx = np.argsort(-df_deg['waldStat'])
    df_deg = df_deg.iloc[idx[:n_deg_genes]]
    yhat_scaled = df_yhat.loc[df_deg.index]
    x_pca = PCA(n_components=deg_clustering_npc, svd_solver='full').fit_transform(yhat_scaled.values)
    cluster_labels = leiden_clustering(
        x_pca, 
        k=deg_clustering_knn, 
        resolution=deg_clustering_res, 
        input='embedding')

    data_tmp = np.concatenate((df_deg.values, cluster_labels.reshape(-1,1)),axis=1)
    df_metadata = pd.DataFrame(data=data_tmp, index=df_deg.index,
        columns=['waldStat','df','pvalue','cluster'] )
    return df_metadata, yhat_scaled

def communication_responseGenes_keggEnrich(
    gene_list: list = None,
    gene_sets: str = "KEGG_2021_Human",
    organism: str = "Human"
):
    """
    Function for performing KEGG enrichment analysis on a given list of response genes.

    Parameters
    ----------
    gene_list
        A list of genes to be analyzed for enrichment. Default is None.
    gene_sets
        The gene set database to use for enrichment analysis. Default is "KEGG_2021_Human".
        For mouse, use 'KEGG_2019_Mouse'.
    organism
        The organism for which the gene sets are defined. Default is "Human".
        For mouse, use 'Mouse'.

    Returns
    -------
    df_result : pandas.DataFrame
        A DataFrame containing the results of the enrichment analysis.
    """

    enr = gp.enrichr(gene_list = gene_list,
                     gene_sets = gene_sets,
                     organism = organism,
                     no_plot = True,
                     cutoff = 0.5)
    df_result = enr.results
    
    return df_result

def compute_direction_histogram_per_pair(
    adata: anndata.AnnData,
    database_name: str,
    all_ms_pairs: list,
    summary: str = "receiver",
    grid_density: float = 0.5,
    n_bins: int = 18,
    eps: float = 1e-3
):
    """
    Compute per-pair directional histograms for MCC vector fields.

    This function computes a 19-dimensional direction histogram for each
    metabolite–sensor (M–S) pair, based on the angular distribution of
    local communication flow vectors. Grid points that are zero vectors
    across all pairs are removed before histogramming.

    The last bin (``dir_bin_zero``) represents the fraction of zero vectors,
    while the remaining ``n_bins`` bins capture the normalized angular
    distribution of nonzero vectors.

    Steps
    -----
    1. For each M–S pair, retrieve the vector field using 
       :func:`mc.pl.plot_communication_flow` (grid mode).
    2. Stack all vector fields and identify grid positions that are zero across all pairs.
    3. Remove all-zero grids and retain only informative ones.
    4. For each remaining M–S pair:
       - Normalize vector magnitudes by their maximum norm.
       - Set vectors below ``eps`` to zero.
       - Compute the histogram of angles using ``atan2(Vy, Vx)`` with ``n_bins`` bins.
       - Compute the zero fraction (ratio of zero vectors to total grid points).
       - Concatenate ``n_bins`` angular bins and one zero bin (total 19 features).
    5. Return a DataFrame summarizing per-pair histograms and the filtered grid coordinates.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing spatial coordinates and MCC flow results.
    database_name : str
        Name of the metabolite–sensor database used in MCC inference.
    all_ms_pairs : list
        List of metabolite–sensor pair names to process.
    summary : str, default="receiver"
        Whether to summarize the flow field from the "sender" or "receiver" perspective.
    grid_density : float, default=0.5
        Density parameter controlling the grid resolution for vector field sampling.
    n_bins : int, default=18
        Number of angular bins for histogramming directions.
    eps : float, default=1e-3
        Threshold below which vectors are treated as zero.

    Returns
    -------
    df_hist : pandas.DataFrame
        DataFrame of shape (n_pairs × (n_bins + 1)) containing the angular
        distribution (``dir_bin_0``–``dir_bin_{n_bins-1}``) and zero fraction
        (``dir_bin_zero``) for each metabolite–sensor pair.
    coords_filtered : numpy.ndarray
        Array of filtered grid coordinates (G × 2) retained after removing
        all-zero grid points.

    Notes
    -----
    - This function is designed to produce direction histograms that
      summarize flow orientation patterns across multiple metabolite–sensor pairs.
    - The output is suitable for downstream analyses such as clustering or
      comparing flow directionality patterns between pathways or tissue regions.
    - The function internally suppresses plotting figures for performance.
    """

    V_all = []
    coords_ref = None
    valid_pairs = []

    for ms_pair_name in all_ms_pairs:
        fig, ax = plt.subplots(figsize=(4, 4))
        _, coords_plot, V_plot = plot_communication_flow(
            adata=adata,
            database_name=database_name,
            ms_pair_name=ms_pair_name,
            summary=summary,
            plot_method="grid",
            background="image",
            grid_density=grid_density,
            normalize_v_quantile=0.995,
            ax=ax
        )
        plt.close(fig)

        if coords_ref is None:
            coords_ref = coords_plot.copy()

        V_all.append(V_plot)
        valid_pairs.append(ms_pair_name)

    if len(V_all) == 0:
        print("All flows are empty.")
        return pd.DataFrame(), None

    V_all = np.stack(V_all, axis=0)  # (n_pairs, G, 2)

    all_norms = np.linalg.norm(V_all, axis=2)
    keep_mask = ~(np.all(all_norms < 1e-12, axis=0))
    kept_indices = np.where(keep_mask)[0]
    coords_filtered = coords_ref[kept_indices, :]
    print(f"Filtered out {np.sum(~keep_mask)} all-zero grids, kept {len(kept_indices)}.")

    df_rows = []
    for i, ms_pair_name in enumerate(valid_pairs):
        V = V_all[i, keep_mask, :]
        norms = np.linalg.norm(V, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0:
            Vn = V / max_norm
            norms_n = norms / max_norm
        else:
            Vn = V.copy()
            norms_n = norms.copy()

        zero_mask = norms_n < eps
        n_zero = np.sum(zero_mask)
        n_nonzero = np.sum(~zero_mask)

        # --- angles of nonzero vectors ---
        if n_nonzero > 0:
            angles = np.arctan2(Vn[~zero_mask, 1], Vn[~zero_mask, 0])
            counts, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
            # Normalize angular bins to sum = 1 over *nonzero* vectors
            prob_angular = counts.astype(float) / n_nonzero
        else:
            prob_angular = np.zeros(n_bins)

        # --- zero fraction (still out of all grids) ---
        zero_frac = n_zero / (n_zero + n_nonzero)

        # --- concatenate 18 angular bins + 1 zero bin ---
        hist19 = np.concatenate([prob_angular, [zero_frac]])
        df_rows.append(hist19)

    df_hist = pd.DataFrame(
        df_rows,
        index=valid_pairs,
        columns=[f"dir_bin_{i}" for i in range(n_bins)] + ["dir_bin_zero"]
    )

    print(f"Computed {len(valid_pairs)} flows × {n_bins+1} bins (after filtering).")
    return df_hist, coords_filtered