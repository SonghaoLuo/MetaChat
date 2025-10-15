import anndata
import random
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import gseapy as gp
from tqdm import tqdm
import networkx as nx
from scipy import sparse
from typing import Optional
from collections import Counter
from multiprocessing import Pool
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
    df_metasen = adata.uns["df_metasen_filtered"]
    
    # Summary by specific metabolites
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

    # Summary by specific metabolic pathway
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
    
    # Summary by specific customer list
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

################## MCC flow ##################
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

    obsp_names_sender.append(database_name+'-sender-total-total')
    obsp_names_receiver.append(database_name+'-receiver-total-total')
    if sum_metabolites is not None and sum_metapathways is not None and sum_customerlists is not None:
        print("Neither sum_metabolites, sum_metapathways, sum_customerlists are provided, just calculate MCC for all signals")

    pts = np.array( adata.obsm[spatial_key], float )
    if not pos_idx is None:
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

        # # ############################## try1 #################################
        # sender_vf = np.zeros_like(pts)
        # receiver_vf = np.zeros_like(pts)

        # S_lil = P_sender.tolil()
        # for j in range(P_sender.shape[0]):
        #     tmp_idx = np.array( S_lil.rows[j], int )
        #     tmp_data = np.array( S_lil.data[j], float )

        #     if len(tmp_idx) == 0:
        #         continue
        #     elif len(tmp_idx) != 0:
        #         yj = pts[tmp_idx,:]
        #         yj_ =  np.sum(yj * tmp_data.reshape(-1,1))/ np.sum(tmp_data.reshape(-1,1)) 
        #     sender_vf[j,:] = yj_ - pts[j,:]
        
        # S_lil = P_receiver.T.tolil()
        # for j in range(P_receiver.shape[1]):
        #     tmp_idx = np.array( S_lil.rows[j], int )
        #     tmp_data = np.array( S_lil.data[j], float )

        #     if len(tmp_idx) == 0:
        #         continue
        #     elif len(tmp_idx) != 0:
        #         yj = pts[tmp_idx,:]
        #         yj_ =  np.sum(yj * tmp_data.reshape(-1,1))/ np.sum(tmp_data.reshape(-1,1)) 
        #     receiver_vf[j,:] = pts[j,:] - yj_

        # adata.obsm["MetaChat-vf-"+obsp_names_sender[i]] = sender_vf
        # adata.obsm["MetaChat-vf-"+obsp_names_receiver[i]] = receiver_vf
        
        # # ############################## try2 #################################
        # MCC_vf = np.zeros_like(pts, dtype=float)

        # S_sent = P_sender.tolil()       # rows: i->* (outflow from i)
        # S_received = P_sender.T.tolil() # rows: j<-* (inflow to j)
        # Tii = np.asarray(P_sender.diagonal(), dtype=float)  # self-loop strengths

        # n = P_sender.shape[0]
        # for j in range(n):

        #     # -------- Outflow resultant from cell j: sum_i T[j,i] * unit(j->i) --------
        #     cols = np.array(S_sent.rows[j], dtype=int)      # indices i where T[j,i] != 0
        #     if cols.size > 0:
        #         # direction j->i
        #         vec_out = pts[cols, :] - pts[j, :]          # (m, 2)
        #         # normalize rows to unit vectors
        #         u_out = normalize(vec_out, norm='l2', axis=1)
        #         w_out = np.asarray(S_sent.data[j], dtype=float)  # (m,)
        #         v_out = (u_out * w_out[:, None]).sum(axis=0)     # (2,)
        #     else:
        #         v_out = np.zeros(2, dtype=float)

        #     # -------- Inflow resultant to cell j: sum_i T[i,j] * unit(i->j) --------
        #     rows = np.array(S_received.rows[j], dtype=int)  # indices i where T[i,j] != 0
        #     if rows.size > 0:
        #         # direction i->j  (note: vec_in = j - i)
        #         vec_in = pts[j, :] - pts[rows, :]           # (m, 2)
        #         u_in = normalize(vec_in, norm='l2', axis=1)
        #         w_in = np.asarray(S_received.data[j], dtype=float)  # (m,)
        #         v_in = (u_in * w_in[:, None]).sum(axis=0)            # (2,)
        #     else:
        #         v_in = np.zeros(2, dtype=float)

        #     # -------- Self-loop cancellation vector from Tii --------
        #     # Normalize inflow resultant to get its direction; if zero, no cancellation.
        #     norm_in = float(np.linalg.norm(v_in))
        #     if norm_in > 0:
        #         in_dir = v_in / norm_in
        #         # coefficient: (Tii - ||in||), clipped at 0 to avoid reversing direction
        #         coeff = max(0, Tii[j] - norm_in)
        #         v_self_cancel = in_dir * coeff
        #     else:
        #         v_self_cancel = np.zeros(2, dtype=float)

        #     # -------- Final per-cell vector --------
        #     MCC_vf[j, :] = v_out - v_self_cancel

        # adata.obsm["MetaChat-vf-"+obsp_names_sender[i]] = MCC_vf
        # adata.obsm["MetaChat-vf-"+obsp_names_receiver[i]] = MCC_vf
                
        ####################################################################

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

def init_communication_group(_adata):
    global adata
    adata = _adata

def _compute_group_result(args):
    group_name, clusterid, celltypes, summary, obsp_name, n_permutations = args
    key = 'MetaChat-' + obsp_name
    S = adata.obsp[key]
    tmp_df, tmp_p_value = summarize_group(S, clusterid, celltypes, n_permutations)
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
    
    if sum_metabolites is None and sum_metapathways is None and sum_customerlists is None and sum_ms_pairs is None:
        print("None of sum_metabolites, sum_metapathways, sum_customerlists, or sum_ms_pairs is provided. Just calculate group-level MCC for all signals.")
    # Check keys
    for i in range(len(obsp_names)):
        key = 'MetaChat-'+obsp_names[i]
        if not key in adata.obsp.keys():
            raise KeyError(f"Please check whether the mc.tl.summary_communication function run or whether {key} are in adata.obsp.keys().")

    task_list = [(group_name, clusterid, celltypes, summary, name, n_permutations) for name in obsp_names]
    
    results = []
    if use_parallel:
        with Pool(processes=n_jobs, initializer=init_communication_group, initargs=(adata,)) as pool:
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

def init_spatial_permutation(_S_list, _bin_positions, _index_obsp_list, _bin_counts_ij, _bin_total_counts_ij):

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

    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."

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

    if sum_metabolites is None and sum_metapathways is None and sum_customerlists is None and sum_ms_pairs is None:
        print("None of sum_metabolites, sum_metapathways, sum_customerlists, or sum_ms_pairs is provided. Just calculate group-level MCC for all signals.")

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
        with Pool(processes=n_jobs, initializer=init_spatial_permutation,
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

################## MCC pathway summary ##################
def summary_pathway(adata: anndata.AnnData,
                    database_name: str = None,
                    group_name: str = None,
                    summary: str = 'sender',
                    sender_group: str = None,
                    receiver_group: str = None,
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

################## MCC remodelling ##################
def communication_responseGenes(
    adata: anndata.AnnData,
    adata_raw: anndata.AnnData,
    database_name: str = None,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    group_name: str = None,
    subgroup: list = None,
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

    # if n_var_genes is None:
    #     sc.pp.highly_variable_genes(adata_deg_var, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # elif not n_var_genes is None:
    #     sc.pp.highly_variable_genes(adata_deg_var, n_top_genes=n_var_genes)
    # if var_genes is None:
    #     adata_deg_raw = adata_deg_raw[:, adata_deg_var.var.highly_variable]
    # else:
    #     adata_deg_raw = adata_deg_raw[:, var_genes]
    # del adata_deg_var

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