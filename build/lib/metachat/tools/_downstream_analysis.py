import anndata
import random
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
from tqdm import tqdm
import networkx as nx
from scipy import sparse
from typing import Optional
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .._utils import leiden_clustering

################## MCC communication summary ##################
def summary_communication(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    copy: bool = False):

    """
    Function for summary communication signals to different metabolites set.

    Parameters
    ----------
    adata
        The AnnData object that have run "mc.tl.metabolic_communication".
        Rows correspond to cells or spots and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    sum_metabolites
        List of specific metabolites to summarize communication for. 
        For example, sum_metabolites = ['HMDB0000148','HMDB0000674'].
    sum_metapathways
        List of specific metabolic pathways to summarize communication for.
        For example, sum_metapathways = ['Alanine, aspartate and glutamate metabolism','Glycerolipid Metabolism'].
    sum_customerlists
        Dictionary of custom lists to summarize communication for. Each key represents a customer name and the value is a list of metabolite-sensor pairs.
        For example, sum_customerlists = {'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')], 'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.

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

    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert sum_metabolites is not None or sum_metapathways is not None or sum_customerlists is not None, "Please ensure that at least one of these three parameters (sum_metabolites, sum_metapathways, sum_customerlists) is given a valid variable."
    
    ncell = adata.shape[0]
    df_metasen = adata.uns["Metabolite_Sensor_filtered"]
    
    # Summary by specific metabolites
    if sum_metabolites is not None:
        S_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metabolites))]
        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_metabolites))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_metabolites))]
        col_names_sender_all = []
        col_names_receiver_all = []
        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_metabolite in range(len(sum_metabolites)):
            metabolite_name = sum_metabolites[idx_metabolite]
            if metabolite_name in df_metasen['Metabolite'].values:
                idx_related = np.where(df_metasen["Metabolite"].str.contains(metabolite_name, na=False))[0]

                for i in idx_related:
                    S = adata.obsp['MetaChat-' + database_name + '-' + df_metasen.loc[i,'Metabolite'] + '-' + df_metasen.loc[i,'Sensor']]
                    S_list[idx_metabolite] = S_list[idx_metabolite] + S
                    X_sender_list[idx_metabolite] = X_sender_list[idx_metabolite] + np.array(S.sum(axis=1))
                    X_receiver_list[idx_metabolite] = X_receiver_list[idx_metabolite] + np.array(S.sum(axis=0).T)
                adata.obsp['MetaChat-' + database_name + '-' + metabolite_name] = S_list[idx_metabolite]
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

    # Summary by specific metabolic pathway
    if sum_metapathways is not None:
        S_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metapathways))]
        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_metapathways))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_metapathways))]
        col_names_sender_all = []
        col_names_receiver_all = []
        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_pathway in range(len(sum_metapathways)):
            pathway_name = sum_metapathways[idx_pathway]
            if np.sum(df_metasen["Metabolite.Pathway"].str.contains(pathway_name, na=False)) > 0:
                idx_related = np.where(df_metasen["Metabolite.Pathway"].str.contains(pathway_name, na=False))[0]

                for i in idx_related:
                    S = adata.obsp['MetaChat-' + database_name + '-' + df_metasen.loc[i,'Metabolite'] + '-' + df_metasen.loc[i,'Sensor']]
                    S_list[idx_pathway] = S_list[idx_pathway] + S
                    X_sender_list[idx_pathway] = X_sender_list[idx_pathway] + np.array(S.sum(axis=1))
                    X_receiver_list[idx_pathway] = X_receiver_list[idx_pathway] + np.array(S.sum(axis=0).T)
                adata.obsp['MetaChat-' + database_name + '-' + pathway_name] = S_list[idx_pathway]
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
    
    # Summary by specific customer list
    if sum_customerlists is not None:
        S_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_customerlists))]
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
                S = adata.obsp['MetaChat-' + database_name + '-' + temp_meta + '-' + temp_sens]
                S_list[idx_customerlist] = S_list[idx_customerlist] + S
                X_sender_list[idx_customerlist] = X_sender_list[idx_customerlist] + np.array(S.sum(axis=1))
                X_receiver_list[idx_customerlist] = X_receiver_list[idx_customerlist] + np.array(S.sum(axis=0).T)     
            adata.obsp['MetaChat-' + database_name + '-' + customerlist_name] = S_list[idx_customerlist]
            X_sender_all = np.concatenate((X_sender_all, X_sender_list[idx_customerlist]), axis=1)
            X_receiver_all = np.concatenate((X_receiver_all, X_receiver_list[idx_customerlist]), axis=1)
            col_names_sender_all.append("s-" + customerlist_name)
            col_names_receiver_all.append("r-" + customerlist_name)

        df_sender_all = pd.DataFrame(data=X_sender_all, columns=col_names_sender_all, index=adata.obs_names)
        df_receiver_all = pd.DataFrame(data=X_receiver_all, columns=col_names_receiver_all, index=adata.obs_names)

        adata.obsm['MetaChat-' + database_name + '-sum-sender-customer'] = df_sender_all
        adata.obsm['MetaChat-' + database_name + '-sum-receiver-customer'] = df_receiver_all

    return adata if copy else None

################## MCC flow ##################
def communication_flow(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    k: int = 5,
    pos_idx: Optional[np.ndarray] = None,
    copy: bool = False
):
    """
    Function for constructing metabolic communication flow by a vector field.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    sum_metabolites
        List of specific metabolites to summarize communication for. 
        For example, sum_metabolites = ['HMDB0000148','HMDB0000674'].
    sum_metapathways
        List of specific metabolic pathways to summarize communication for.
        For example, sum_metapathways = ['Alanine, aspartate and glutamate metabolism','Glycerolipid Metabolism'].
    sum_customerlists
        Dictionary of custom lists to summarize communication for. Each key represents a customer name and the value is a list of metabolite-sensor pairs.
        For example, sum_customerlists = {'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')], 'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}
    k
        Top k senders or receivers to consider when determining the direction.
    pos_idx
        The columns in ``.obsm['spatial']`` to use. If None, all columns are used.
        For example, to use just the first and third columns, set pos_idx to ``numpy.array([0,2],int)``.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        sum_metabolites, sum_metapathways, sum_customerlists can provided by user in one time.  
        Vector fields describing signaling directions are added to ``.obsm``. For example:  
        ``.obsm['MetaChat_sender_vf-databaseX-metA-senA']`` and ``.obsm['MetaChat_receiver_vf-databaseX-metA-senA']``
        For each "metabolite_name" in "sum_metabolites", ``adata.obsm['MetaChat_sender_vf'+database_name+'-'+metabolite_name]`` and ``adata.obsm['MetaChat_receiver_vf'+database_name+'-'+metabolite_name]``.
        For each "pathway_name" in "sum_metapathways", ``adata.obsm['MetaChat_sender_vf'+database_name+'-'+pathway_name]`` and ``adata.obsm['MetaChat_receiver_vf'+database_name+'-'+pathway_name]``.
        For each "customerlist_name" in "sum_customerlists", ``adata.obsm['MetaChat_sender_vf'+database_name+'-'+customerlist_name]`` and ``adata.obsm['MetaChat_receiver_vf'+database_name+'-'+customerlist_name]``.
        If copy=True, return the AnnData object and return None otherwise.

    """
    # Check inputs
    assert database_name is not None, "Please at least specify database_name."

    obsp_names = []
    if sum_metabolites is not None:
        for metabolite_name in sum_metabolites:
            obsp_names.append(database_name + '-' + metabolite_name)
    
    if sum_metapathways is not None:
        for pathway_name in sum_metapathways:
            obsp_names.append(database_name + '-' + pathway_name)

    if sum_customerlists is not None:
        for customerlist_name in sum_customerlists.keys():
            obsp_names.append(database_name + '-' + customerlist_name)

    obsp_names.append(database_name+'-total-total')
    if sum_metabolites is not None and sum_metapathways is not None and sum_customerlists is not None:
        print("Neither sum_metabolites, sum_metapathways, sum_customerlists are provided, just calculate MCC for all signals")

    pts = np.array( adata.obsm['spatial'], float )
    if not pos_idx is None:
        pts = pts[:,pos_idx]

    for i in range(len(obsp_names)):
        key = 'MetaChat-'+obsp_names[i]
        if not key in adata.obsp.keys():
            raise KeyError(f"Please check whether the mc.tl.summary_communication function run or whether {key} are in adata.obsp.keys().")
        S = adata.obsp[key]
        S_sum_sender = np.array( S.sum(axis=1) ).reshape(-1)
        S_sum_receiver = np.array( S.sum(axis=0) ).reshape(-1)
        sender_vf = np.zeros_like(pts)
        receiver_vf = np.zeros_like(pts)

        S_lil = S.tolil()
        for j in range(S.shape[0]):
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
            sender_vf[j,:] = avg_v[0,:] * S_sum_sender[j]
        
        S_lil = S.T.tolil()
        for j in range(S.shape[0]):
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
            receiver_vf[j,:] = avg_v[0,:] * S_sum_receiver[j]

        adata.obsm["MetaChat_sender_vf-"+obsp_names[i]] = sender_vf
        adata.obsm["MetaChat_receiver_vf-"+obsp_names[i]] = receiver_vf

    return adata if copy else None


################## Group-level MCC ##################
def summarize_group(X, clusterid, clusternames, n_permutations=100):
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

def communication_group(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    n_permutations: int = 100,
    random_seed: int = 1,
    copy: bool = False
):
    """
    Function for summarizng metabolic MCC communication to group-level communication and computing p-values by permutating cell/spot labels.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    group_name
        Group name of the cell annotation previously saved in ``adata.obs``. 
    sum_metabolites
        List of specific metabolites to summarize communication for. 
        For example, sum_metabolites = ['HMDB0000148','HMDB0000674'].
    sum_metapathways
        List of specific metabolic pathways to summarize communication for.
        For example, sum_metapathways = ['Alanine, aspartate and glutamate metabolism','Glycerolipid Metabolism'].
    sum_customerlists
        Dictionary of custom lists to summarize communication for. Each key represents a customer name and the value is a list of metabolite-sensor pairs.
        For example, sum_customerlists = {'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')], 'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}
    n_permutations
        Number of label permutations for computing the p-value.
    random_seed
        The numpy random_seed for reproducible random permutations.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        Add group-level communication matrix to ``.uns['MetaChat_group-'+group_name+'-'+database_name+'-'+metabolite_name]``, ``.uns['MetaChat_group-'+group_name+'-'+database_name+'-'+pathway_name]`` or ``.uns['MetaChat_group-'+group_name+'-'+database_name+'-'+customerlist_name]``
        The first key is the communication intensity matrix ['communication_matrix']
        The second key is the p-value ['communication_pvalue'].
        If copy=True, return the AnnData object and return None otherwise.

    """
    np.random.seed(random_seed)

    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."

    celltypes = list( adata.obs[group_name].unique() )
    celltypes.sort()
    for i in range(len(celltypes)):
        celltypes[i] = str(celltypes[i])
    clusterid = np.array(adata.obs[group_name], str)

    obsp_names = []
    if sum_metabolites is not None:
        for metabolite_name in sum_metabolites:
            obsp_names.append(database_name + '-' + metabolite_name)
    
    if sum_metapathways is not None:
        for pathway_name in sum_metapathways:
            obsp_names.append(database_name + '-' + pathway_name)

    if sum_customerlists is not None:
        for customerlist_name in sum_customerlists.keys():
            obsp_names.append(database_name + '-' + customerlist_name)

    obsp_names.append(database_name+'-total-total')
    if sum_metabolites is not None and sum_metapathways is not None and sum_customerlists is not None:
        print("Neither sum_metabolites, sum_metapathways, sum_customerlists are provided, just calculate group-level MCC for all signals")

    for i in range(len(obsp_names)):
        key = 'MetaChat-'+obsp_names[i]
        if not key in adata.obsp.keys():
            raise KeyError(f"Please check whether the mc.tl.summary_communication function run or whether {key} are in adata.obsp.keys().")
        S = adata.obsp['MetaChat-'+obsp_names[i]]
        tmp_df, tmp_p_value = summarize_group(S, clusterid, celltypes, n_permutations=n_permutations)
        adata.uns['MetaChat_group-'+group_name+'-'+obsp_names[i]] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}
    
    return adata if copy else None

def communication_group_spatial(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    sum_metabolites: list = None,
    sum_metapathways: list = None,
    sum_customerlists: dict = None,
    n_permutations: int = 100,
    bins_num: int = 30,
    random_seed: int = 1,
    copy: bool = False):
    
    """
    Function for summarizng metabolic MCC communication to group-level communication and computing p-values based on spaital distance distribution.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    group_name
        Group name of the cell annotation previously saved in ``adata.obs``. 
    sum_metabolites
        List of specific metabolites to summarize communication for. 
        For example, sum_metabolites = ['HMDB0000148','HMDB0000674'].
    sum_metapathways
        List of specific metabolic pathways to summarize communication for.
        For example, sum_metapathways = ['Alanine, aspartate and glutamate metabolism','Glycerolipid Metabolism'].
    sum_customerlists
        Dictionary of custom lists to summarize communication for. Each key represents a customer name and the value is a list of metabolite-sensor pairs.
        For example, sum_customerlists = {'CustomerA': [('HMDB0000148', 'Grm5'), ('HMDB0000148', 'Grm8')], 'CustomerB': [('HMDB0000674', 'Trpc4'), ('HMDB0000674', 'Trpc5')]}
    n_permutations
        Number of label permutations for computing the p-value.
    bins_num
        Number of bins for sampling based on spaital distance distribution.
    random_seed
        The numpy random_seed for reproducible random permutations.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        Add group-level communication matrix to ``.uns['MetaChat_group_spatial-'+group_name+'-'+database_name+'-'+metabolite_name]``, ``.uns['MetaChat_group-'+group_name+'-'+database_name+'-'+pathway_name]`` or ``.uns['MetaChat_group-'+group_name+'-'+database_name+'-'+customerlist_name]``
        The first key is the communication intensity matrix ['communication_matrix']
        The second key is the p-value ['communication_pvalue'].
        If copy=True, return the AnnData object and return None otherwise.

    """

    np.random.seed(random_seed)

    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."

    celltypes = list( adata.obs[group_name].unique() )
    celltypes.sort()
    for i in range(len(celltypes)):
        celltypes[i] = str(celltypes[i])
    clusterid = np.array(adata.obs[group_name], str)

    obsp_names = []
    if sum_metabolites is not None:
        for metabolite_name in sum_metabolites:
            obsp_names.append(database_name + '-' + metabolite_name)

    if sum_metapathways is not None:
        for pathway_name in sum_metapathways:
            obsp_names.append(database_name + '-' + pathway_name)

    if sum_customerlists is not None:
        for customerlist_name in sum_customerlists.keys():
            obsp_names.append(database_name + '-' + customerlist_name)

    obsp_names.append(database_name+'-total-total')
    if sum_metabolites is not None and sum_metapathways is not None and sum_customerlists is not None:
        print("Neither sum_metabolites, sum_metapathways, sum_customerlists are provided, just calculate group-level MCC for all signals")

    dist_matrix = adata.obsp['spatial_distance_LRC_No']
    hist, bin_edges = np.histogram(dist_matrix, bins=bins_num)
    dist_matrix_bin = np.digitize(dist_matrix, bin_edges) - 1
    bin_positions = {category: np.argwhere(dist_matrix_bin == category) for category in range(bins_num + 1)}

    n = len(celltypes)
    bin_counts_ij = [[{} for j in range(n)] for i in range(n)]
    bin_total_counts_ij = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            tmp_idx_i = np.where(clusterid == celltypes[i])[0]
            tmp_idx_j = np.where(clusterid == celltypes[j])[0]
            tmp_idx_bin = dist_matrix_bin[tmp_idx_i,:][:,tmp_idx_j]
            tmp_idx_bin_flatten = tmp_idx_bin.flatten()
            bin_counts_ij[i][j] = Counter(tmp_idx_bin_flatten)
            bin_total_counts_ij[i,j] = len(tmp_idx_i) * len(tmp_idx_j)

    S = {}
    X_cluster = {}
    p_cluster = {}

    for index_obsp in range(len(obsp_names)):

        print('Computing group-level MCC for ' + obsp_names[index_obsp])
        key = 'MetaChat-'+obsp_names[index_obsp]
        if not key in adata.obsp.keys():
            raise KeyError(f"Please check whether the mc.tl.summary_communication function run or whether {key} are in adata.obsp.keys().")
        S[index_obsp] = adata.obsp[key]
        X_cluster_temp = np.empty([n,n], float)

        for i in range(n):
            tmp_idx_i = np.where(clusterid == celltypes[i])[0]
            for j in range(n):
                tmp_idx_j = np.where(clusterid == celltypes[j])[0]
                X_cluster_temp[i,j] = S[index_obsp][tmp_idx_i,:][:,tmp_idx_j].mean()
        X_cluster[index_obsp] = X_cluster_temp
        p_cluster[index_obsp] = np.zeros([n,n], float)
    
    for i, j in tqdm(list(itertools.product(range(n), repeat=2)), desc='Processing pairs'):
        
        X_cluster_permut_ij = {}
        for index_obsp in range(len(obsp_names)):
            X_cluster_permut_ij[index_obsp] = np.zeros(n_permutations)
            
        for trial in range(n_permutations):
            tmp_X_cluster_permut_ij = {}
            for index_obsp in range(len(obsp_names)):
                tmp_X_cluster_permut_ij[index_obsp] = 0

            for bin, count in bin_counts_ij[i][j].items():
                positions = bin_positions[bin]
                sampled_indices = random.sample(range(len(positions)), count)
                sampled_positions = positions[sampled_indices]
                for index_obsp in range(len(obsp_names)):
                    tmp_X_cluster_permut_ij[index_obsp] += S[index_obsp][sampled_positions[:, 0], sampled_positions[:, 1]].sum()
                    X_cluster_permut_ij[index_obsp][trial] = tmp_X_cluster_permut_ij[index_obsp] / bin_total_counts_ij[i,j]
        
        for index_obsp in range(len(obsp_names)):
            p_value = np.sum(np.array(X_cluster_permut_ij[index_obsp]) >= X_cluster[index_obsp][i,j]) / n_permutations
            p_cluster[index_obsp][i,j] = p_value

    for index_obsp in range(len(obsp_names)):
        df_cluster = pd.DataFrame(data = X_cluster[index_obsp], index = celltypes, columns = celltypes)
        df_p_value = pd.DataFrame(data = p_cluster[index_obsp], index = celltypes, columns = celltypes)
        adata.uns['MetaChat_group_spatial-'+group_name+'-'+obsp_names[index_obsp]] = {'communication_matrix': df_cluster, 'communication_pvalue': df_p_value}
    
    return adata if copy else None

################## MCC pathway summary ##################
def summary_pathway(adata: anndata.AnnData,
                    database_name: str = None,
                    group_name: str = None,
                    sender_group: str = None,
                    receiver_group: str = None,
                    usenumber_metapathway: int = 5,
                    permutation_spatial: bool = False):
    """
    Function for summarizng MCC pathway pattern given specific sender group and receiver group.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    group_name
        Group name of the cell annotation previously saved in ``adata.obs``. 
    sender_group
        Name of the sender group
    receiver_group
        Name of the receiver group
    usenumber_metapathway
        Number of top metabolic pathways to use in the summary. Default is 5.
    permutation_spatial
        Whether to use results from ``mc.tl.communication_group_spatial``.
    
    Returns
    -------
    metapathway_rank : pd.DataFrame
        Ranking of metabolic pathways.
    senspathway_rank : pd.DataFrame
        Ranking of sensor's pathways.
    ms_result : pd.DataFrame
        The data frame of communication intensity between meatbolic pathway and sensor pathway.
    """
    
    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."
    assert sender_group is not None, "Please at least specify sender_group."
    assert receiver_group is not None, "Please at least specify receiver_group."

    df_metasen = adata.uns["Metabolite_Sensor_filtered"].copy()
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
            key = "MetaChat_group_spatial-" + group_name + "-" + database_name + "-" + pathway_name
            if not key in adata.uns.keys():
                raise KeyError(f"Please check whether the mc.tl.communication_group_spatial function are run and whether {key} are in adata.uns.keys()." \
                               "Note that this function needs to compute the group-level for all pathways")
            MCC_metapathway.loc[pathway_name,"communication_score"] = adata.uns[key]["communication_matrix"].loc[sender_group,receiver_group]
            MCC_metapathway.loc[pathway_name,"p_value"] = adata.uns[key]["communication_pvalue"].loc[sender_group,receiver_group]
        else:
            key = "MetaChat_group-" + group_name + "-" + database_name + "-" + pathway_name
            if not key in adata.uns.keys():
                raise KeyError(f"Please check whether the mc.tl.communication_group function are run and whether {key} are in adata.uns.keys()." \
                               "Note that this function needs to compute the group-level for all pathways")
            MCC_metapathway.loc[pathway_name,"communication_score"] = adata.uns[key]["communication_matrix"].loc[sender_group,receiver_group]
            MCC_metapathway.loc[pathway_name,"p_value"] = adata.uns[key]["communication_pvalue"].loc[sender_group,receiver_group]
      
    metapathway_rank = MCC_metapathway.sort_values(by=['p_value', 'communication_score'], ascending=[True, False])
    use_metapatheway = metapathway_rank.index[:usenumber_metapathway,].tolist()

    # Compute the each m-s pairs communication_score
    MCC_group_pair = adata.uns['Metabolite_Sensor_filtered'].copy()
    for irow, ele in MCC_group_pair.iterrows():
        Metaname = ele['Metabolite']
        Sensname = ele['Sensor']
        key = "MetaChat_group-" + group_name + "-" + database_name + "-" + Metaname + "-" + Sensname
        if not key in adata.uns.keys():
                raise KeyError(f"Please check whether the mc.tl.communication_group function are run and whether {key} are in adata.uns.keys()." \
                               "Note that this function needs to compute the group-level for all m-s pairs")
        MCC_group_pair.loc[irow, "communication_score"] = adata.uns[key]["communication_matrix"].loc[sender_group,receiver_group]

    MCC_Meta2pathway = MCC_group_pair[["Metabolite", "Metabolite.Pathway", "Sensor", "Sensor.Pathway", "communication_score"]]
    MCC_Meta2pathway = MCC_Meta2pathway[((MCC_Meta2pathway['Metabolite.Pathway'] != "nan") & (MCC_Meta2pathway['Sensor.Pathway'] != "nan"))]
    MCC_Meta2pathway['Metabolite.Pathway'] = MCC_Meta2pathway['Metabolite.Pathway'].str.split('; ')
    MCC_Meta2pathway_expanded1 = MCC_Meta2pathway.explode('Metabolite.Pathway')
    MCC_Meta2pathway_expanded1['Sensor.Pathway'] = MCC_Meta2pathway_expanded1['Sensor.Pathway'].str.split('; ')
    MCC_Meta2pathway_expanded2 = MCC_Meta2pathway_expanded1.explode('Sensor.Pathway')
    MCC_Meta2pathway_group = MCC_Meta2pathway_expanded2.groupby(['Metabolite.Pathway', 'Sensor.Pathway'], as_index=False).agg({'communication_score': 'sum'})
    filtered_MCC_Meta2pathway_group = MCC_Meta2pathway_group[MCC_Meta2pathway_group["Metabolite.Pathway"].isin(use_metapatheway)]

    # construct graph network to measure importance
    G = nx.DiGraph()
    edges_with_weights = [
        (row['Metabolite.Pathway'], row['Sensor.Pathway'], row['communication_score']) 
        for _, row in filtered_MCC_Meta2pathway_group.iterrows()
    ]
    for edge in edges_with_weights:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    hubs, authorities = nx.hits(G, max_iter=500, normalized=True)
    senspathway_rank = sorted(authorities.items(), key=lambda item: item[1], reverse=True)
    senspathway_rank = pd.DataFrame(senspathway_rank, columns=['Senspathway', 'Rankscore'])
    senspathway_rank = senspathway_rank[senspathway_rank['Senspathway'].str.startswith('WP')]

    ms_result = filtered_MCC_Meta2pathway_group.pivot_table(index='Metabolite.Pathway', columns='Sensor.Pathway', values='communication_score')
    ms_result = ms_result.fillna(0)

    return metapathway_rank, senspathway_rank, ms_result

################## MCC remodelling ##################
def communication_responseGenes(
    adata: anndata.AnnData,
    adata_raw: anndata.AnnData,
    database_name: str = None,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    summary: str = 'receiver',
    n_var_genes: int = None,
    var_genes = None,
    n_deg_genes: int = None,
    nknots: int = 6,
    n_points: int = 50,
    deg_pvalue_cutoff: float = 0.05,
):
    """
    Function for identifying signals dependent genes

    Parameters
    ----------
    adata
        adata.AnnData object after running inference function ``mc.tl.metabolic_communication``.
    adata_raw
        adata.AnnData object with raw spatial transcriptome data.
    database_name
        Name of the Metabolite-Sensor interaction database.
    metabolite_name
        Name of a specific metabolite to detect response genes. For example, metabolite_name = 'HMDB0000148'.
    metapathway_name
        Name of a specific metabolic pathways to detect response genes. For example, metabolite_name = 'Alanine, aspartate and glutamate metabolism'.
    customerlist_name
        Name of a specific customerlist to detect response genes. For example, customerlist_name = 'CustomerA'.
    summary
        'sender' or 'receiver'
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
    sc.pp.normalize_total(adata_deg_var, target_sum=1e4)
    sc.pp.log1p(adata_deg_var)
    if n_var_genes is None:
        sc.pp.highly_variable_genes(adata_deg_var, min_mean=0.0125, max_mean=3, min_disp=0.5)
    elif not n_var_genes is None:
        sc.pp.highly_variable_genes(adata_deg_var, n_top_genes=n_var_genes)
    if var_genes is None:
        adata_deg_raw = adata_deg_raw[:, adata_deg_var.var.highly_variable]
    else:
        adata_deg_raw = adata_deg_raw[:, var_genes]
    del adata_deg_var

    if summary == 'sender':
        summary_abbr = 's'
    else:
        summary_abbr = 'r'

    non_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name])
    if non_none_count > 1:
        raise ValueError("Only one of 'metabolite_name', 'metapathway_name', or 'customerlist_name' can be specified.")
    
    if metabolite_name is None and metapathway_name is None and customerlist_name is None:
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

    comm_sum = adata.obsm['MetaChat-' + database_name + "-sum-" + summary + obsm_name][summary_abbr + '-' + sum_name].values.reshape(-1,1)
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
    # ro.r('assoRes[is.nan(assoRes[,"waldStat_1"]),"waldStat_1"] <- 0.0')
    # ro.r('assoRes[is.nan(assoRes[,"df_1"]),"df_1"] <- 0.0')
    # ro.r('assoRes[is.nan(assoRes[,"pvalue_1"]),"pvalue_1"] <- 1.0')
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
    df_deg
        The deg analysis summary data frame obtained by running ``ml.tl.communication_response_genes``.
        Each row corresponds to one tested genes and columns include "waldStat" (Wald statistics), "df" (degrees of freedom), and "pvalue" (p-value of the Wald statistics).
    df_yhat
        The fitted (smoothed) gene expression pattern obtained by running ``ml.tl.communication_responseGenes``.
    deg_clustering_npc
        Number of PCs when performing PCA to cluster gene expression patterns
    deg_clustering_knn
        Number of neighbors when constructing the knn graph for leiden clustering.
    deg_clustering_res
        The resolution parameter for leiden clustering.
    n_deg_genes
        Number of top deg genes to cluster.
    p_value_cutoff
        The p-value cutoff for genes to be included in clustering analysis.

    Returns
    -------
    df_deg_clus: pd.DataFrame
        A data frame of clustered genes.
    df_yhat_clus: pd.DataFrame
        The fitted gene expression patterns of the clustered genes

    """
    df_deg = df_deg[df_deg['pvalue'] <= p_value_cutoff]
    n_deg_genes = min(n_deg_genes, df_deg.shape[0])
    idx = np.argsort(-df_deg['waldStat'])
    df_deg = df_deg.iloc[idx[:n_deg_genes]]
    yhat_scaled = df_yhat.loc[df_deg.index]
    x_pca = PCA(n_components=deg_clustering_npc, svd_solver='full').fit_transform(yhat_scaled.values)
    cluster_labels = leiden_clustering(x_pca, k=deg_clustering_knn, resolution=deg_clustering_res, input='embedding')

    data_tmp = np.concatenate((df_deg.values, cluster_labels.reshape(-1,1)),axis=1)
    df_metadata = pd.DataFrame(data=data_tmp, index=df_deg.index,
        columns=['waldStat','df','pvalue','cluster'] )
    return df_metadata, yhat_scaled

def communication_responseGenes_keggEnrich(
    gene_list: list = None,
    gene_sets: str = "KEGG_2021_Human",
    organism: str = "Human"):

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