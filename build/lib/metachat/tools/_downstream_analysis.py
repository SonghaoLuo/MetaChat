from typing import Optional
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
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .._utils import leiden_clustering

################## MCC communication summary ##################
def summary_communication(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolite: list = None,
    sum_metapathway: list = None,
    sum_customerlist: dict = None
):
    ### This part is expalin how to summary communication in a specific context
    # Except single M-S pair resulted from MetaChat, we have three extra choice to summary
    # communication including specific metabolites related, specific metabolic pathways related, and customer list.
    # As an example:
    # adata = adata_combined_infer_LRC
    # database_name = "MetaChatDB"
    # sum_metabolite = ['HMDB0000148','HMDB0000674']
    # sum_metapathway = ['Alanine, aspartate and glutamate metabolism','Glycerolipid Metabolism']
    # sum_customerlist = {'CustomerA': [('HMDB0000148', 'Grm8'), ('HMDB0000674', 'Trpc4')],
    #                    'CustomerB': [('HMDB0000148', 'Grm5'), ('HMDB0000674', 'Trpc5')]}
    assert database_name is not None, "Please at least specify database_name."

    ncell = adata.shape[0]
    df_MetaSen = adata.uns["Metabolite_Sensor_filtered"]

    assert sum_metabolite is not None or sum_metapathway is not None or sum_customerlist is not None, "Please ensure that at least one of these three parameters (sum_metabolites, sum_metapathway, sum_customerlist) is given a valid variable."

    if sum_metabolite is not None:

        S_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metabolite))]
        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_metabolite))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_metabolite))]
        col_names_sender_all = []
        col_names_receiver_all = []
        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_metabolite in range(len(sum_metabolite)):
            name_metabolite = sum_metabolite[idx_metabolite]
            if name_metabolite in df_MetaSen['Metabolite'].values:
                idx_related = np.where(df_MetaSen["Metabolite"].str.contains(name_metabolite, na=False))[0]

                for i in idx_related:
                    S = adata.obsp['MetaChat-' + database_name + '-' + df_MetaSen.loc[i,'Metabolite'] + '-' + df_MetaSen.loc[i,'Sensor']]
                    S_list[idx_metabolite] = S_list[idx_metabolite] + S
                    X_sender_list[idx_metabolite] = X_sender_list[idx_metabolite] + np.array(S.sum(axis=1))
                    X_receiver_list[idx_metabolite] = X_receiver_list[idx_metabolite] + np.array(S.sum(axis=0).T)
                adata.obsp['MetaChat-' + database_name + '-' + name_metabolite] = S_list[idx_metabolite]
                X_sender_all = np.concatenate((X_sender_all, X_sender_list[idx_metabolite]), axis=1)
                X_receiver_all = np.concatenate((X_receiver_all, X_receiver_list[idx_metabolite]), axis=1)

                col_names_sender_all.append("s-" + name_metabolite)
                col_names_receiver_all.append("r-" + name_metabolite)
            else:
                print(f"Warning: {name_metabolite} is not in the results")

        df_sender_all = pd.DataFrame(data=X_sender_all, columns=col_names_sender_all, index=adata.obs_names)
        df_receiver_all = pd.DataFrame(data=X_receiver_all, columns=col_names_receiver_all, index=adata.obs_names)

        adata.obsm['MetaChat-' + database_name + '-sum-sender-metabolite'] = df_sender_all
        adata.obsm['MetaChat-' + database_name + '-sum-receiver-metabolite'] = df_receiver_all


    if sum_metapathway is not None:

        S_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_metapathway))]
        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_metapathway))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_metapathway))]
        col_names_sender_all = []
        col_names_receiver_all = []
        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_pathway in range(len(sum_metapathway)):
            name_pathway = sum_metapathway[idx_pathway]
            if np.sum(df_MetaSen["Metapathway"].str.contains(name_pathway, na=False)) > 0:
                idx_related = np.where(df_MetaSen["Metapathway"].str.contains(name_pathway, na=False))[0]

                for i in idx_related:
                    S = adata.obsp['MetaChat-' + database_name + '-' + df_MetaSen.loc[i,'Metabolite'] + '-' + df_MetaSen.loc[i,'Sensor']]
                    S_list[idx_pathway] = S_list[idx_pathway] + S
                    X_sender_list[idx_pathway] = X_sender_list[idx_pathway] + np.array(S.sum(axis=1))
                    X_receiver_list[idx_pathway] = X_receiver_list[idx_pathway] + np.array(S.sum(axis=0).T)
                adata.obsp['MetaChat-' + database_name + '-' + name_pathway] = S_list[idx_pathway]
                X_sender_all = np.concatenate((X_sender_all, X_sender_list[idx_pathway]), axis=1)
                X_receiver_all = np.concatenate((X_receiver_all, X_receiver_list[idx_pathway]), axis=1)

                col_names_sender_all.append("s-" + name_pathway)
                col_names_receiver_all.append("r-" + name_pathway)
            else:
                print(f"Warning: {name_pathway} is not in the results")

        df_sender_all = pd.DataFrame(data=X_sender_all, columns=col_names_sender_all, index=adata.obs_names)
        df_receiver_all = pd.DataFrame(data=X_receiver_all, columns=col_names_receiver_all, index=adata.obs_names)

        adata.obsm['MetaChat-' + database_name + '-sum-sender-pathway'] = df_sender_all
        adata.obsm['MetaChat-' + database_name + '-sum-receiver-pathway'] = df_receiver_all

    if sum_customerlist is not None:

        S_list = [sparse.csr_matrix((ncell, ncell), dtype=float) for i in range(len(sum_customerlist))]
        X_sender_list = [np.zeros([ncell,1], float) for i in range(len(sum_customerlist))]
        X_receiver_list = [np.zeros([ncell,1], float) for i in range(len(sum_customerlist))]
        col_names_sender_all = []
        col_names_receiver_all = []
        X_sender_all = np.empty([ncell,0], float)
        X_receiver_all = np.empty([ncell,0], float)

        for idx_customerlist, (name_customerlist, value_customerlist) in enumerate(sum_customerlist.items()):

            for idx_value in value_customerlist:
                temp_meta = idx_value[0]
                temp_sens = idx_value[1]
                S = adata.obsp['MetaChat-' + database_name + '-' + temp_meta + '-' + temp_sens]
                S_list[idx_customerlist] = S_list[idx_customerlist] + S
                X_sender_list[idx_customerlist] = X_sender_list[idx_customerlist] + np.array(S.sum(axis=1))
                X_receiver_list[idx_customerlist] = X_receiver_list[idx_customerlist] + np.array(S.sum(axis=0).T)     
            adata.obsp['MetaChat-' + database_name + '-' + name_customerlist] = S_list[idx_customerlist]
            X_sender_all = np.concatenate((X_sender_all, X_sender_list[idx_customerlist]), axis=1)
            X_receiver_all = np.concatenate((X_receiver_all, X_receiver_list[idx_customerlist]), axis=1)
            col_names_sender_all.append("s-" + name_customerlist)
            col_names_receiver_all.append("r-" + name_customerlist)

        df_sender_all = pd.DataFrame(data=X_sender_all, columns=col_names_sender_all, index=adata.obs_names)
        df_receiver_all = pd.DataFrame(data=X_receiver_all, columns=col_names_receiver_all, index=adata.obs_names)

        adata.obsm['MetaChat-' + database_name + '-sum-sender-customer'] = df_sender_all
        adata.obsm['MetaChat-' + database_name + '-sum-receiver-customer'] = df_receiver_all





################## MCC flow ##################
def communication_flow(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolite: list = None,
    sum_metapathway: list = None,
    sum_customerlist: dict = None,
    k: int = 5,
    pos_idx: Optional[np.ndarray] = None,
    copy: bool = False
):
    """
    Construct spatial vector fields for inferred communication.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    database_name
        Name of the ligand-receptor database.
        If both pathway_name and lr_pair are None, the signaling direction of all ligand-receptor pairs is computed.
    pathway_name
        Name of the signaling pathway.
        If given, only the signaling direction of this signaling pathway is computed.
    lr_pair
        A tuple of ligand-receptor pair. 
        If given, only the signaling direction of this pair is computed.
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
        Vector fields describing signaling directions are added to ``.obsm``, 
        e.g., for a database named "databaseX", 
        ``.obsm['commot_sender_vf-databaseX-ligA-recA']`` and ``.obsm['commot_receiver_vf-databaseX-ligA-recA']``
        describe the signaling directions of the cells as, respectively, senders and receivers through the 
        ligand-receptor pair ligA and recA.
        If copy=True, return the AnnData object and return None otherwise.

    """
    assert database_name is not None, "Please at least specify database_name."
    
    obsp_names = []
    if sum_metabolite is not None:
        for name_metabolite in sum_metabolite:
            obsp_names.append(database_name + '-' + name_metabolite)
    
    if sum_metapathway is not None:
        for name_pathway in sum_metapathway:
            obsp_names.append(database_name + '-' + name_pathway)

    if sum_customerlist is not None:
        for name_customerlist in sum_customerlist.keys():
            obsp_names.append(database_name + '-' + name_customerlist)

    if sum_metabolite is not None and sum_metapathway is not None and sum_customerlist is not None:
        obsp_names.append(database_name+'-total-total')

    pts = np.array( adata.obsm['spatial'], float )
    if not pos_idx is None:
        pts = pts[:,pos_idx]

    for i in range(len(obsp_names)):

        S = adata.obsp['MetaChat-'+obsp_names[i]]
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
def summarize_cluster(X, clusterid, clusternames, n_permutations=500):
    # Input a sparse matrix of cell signaling and output a pandas dataframe
    # for cluster-cluster signaling
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

def cluster_communication(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolite: list = None,
    sum_metapathway: list = None,
    sum_customerlist: dict = None,
    clustering: str = None,
    n_permutations: int = 100,
    random_seed: int = 1,
    copy: bool = False
):
    """
    Summarize cell-cell communication to cluster-cluster communication and compute p-values by permutating cell/spot labels.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the ligand-receptor database.
        If both pathway_name and lr_pair are None, the cluster signaling through all ligand-receptor pairs is summarized.
    pathway_name
        Name of the signaling pathway.
        If given, the signaling through all ligand-receptor pairs of the given pathway is summarized.
    lr_pair
        A tuple of ligand-receptor pair. 
        If given, only the cluster signaling through this pair is computed.
    clustering
        Name of clustering with the labels stored in ``.obs[clustering]``.
    n_permutations
        Number of label permutations for computing the p-value.
    random_seed
        The numpy random_seed for reproducible random permutations.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        Add cluster-cluster communication matrix to 
        ``.uns['commot_cluster-databaseX-clustering-ligA-recA']``
        for the ligand-receptor database named 'databaseX' and the cell clustering 
        named 'clustering' through the ligand-receptor pair 'ligA' and 'recA'.
        The first object is the communication score matrix and the second object contains
        the corresponding p-values.
        If copy=True, return the AnnData object and return None otherwise.

    """
    np.random.seed(random_seed)

    assert database_name is not None, "Please at least specify database_name."

    celltypes = list( adata.obs[clustering].unique() )
    celltypes.sort()
    for i in range(len(celltypes)):
        celltypes[i] = str(celltypes[i])
    clusterid = np.array(adata.obs[clustering], str)

    obsp_names = []
    if sum_metabolite is not None:
        for name_metabolite in sum_metabolite:
            obsp_names.append(database_name + '-' + name_metabolite)
    
    if sum_metapathway is not None:
        for name_pathway in sum_metapathway:
            obsp_names.append(database_name + '-' + name_pathway)

    if sum_customerlist is not None:
        for name_customerlist in sum_customerlist.keys():
            obsp_names.append(database_name + '-' + name_customerlist)

    if sum_metabolite is not None and sum_metapathway is not None and sum_customerlist is not None:
        obsp_names.append(database_name+'-total-total')

    for i in range(len(obsp_names)):
        S = adata.obsp['MetaChat-'+obsp_names[i]]
        tmp_df, tmp_p_value = summarize_cluster(S, clusterid, celltypes, n_permutations=n_permutations)
        adata.uns['MetaChat_cluster-'+clustering+'-'+obsp_names[i]] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}
    
    return adata if copy else None

def cluster_communication_spatial(
    adata: anndata.AnnData,
    database_name: str = None,
    sum_metabolite: list = None,
    sum_metapathway: list = None,
    sum_customerlist: dict = None,
    clustering: str = None,
    n_permutations: int = 100,
    bins_num: int = 30,
    random_seed: int = 1,
    copy: bool = False
):
    np.random.seed(random_seed)

    assert database_name is not None, "Please at least specify database_name."

    celltypes = list( adata.obs[clustering].unique() )
    celltypes.sort()
    for i in range(len(celltypes)):
        celltypes[i] = str(celltypes[i])
    clusterid = np.array(adata.obs[clustering], str)

    obsp_names = []
    if sum_metabolite is not None:
        for name_metabolite in sum_metabolite:
            obsp_names.append(database_name + '-' + name_metabolite)

    if sum_metapathway is not None:
        for name_pathway in sum_metapathway:
            obsp_names.append(database_name + '-' + name_pathway)

    if sum_customerlist is not None:
        for name_customerlist in sum_customerlist.keys():
            obsp_names.append(database_name + '-' + name_customerlist)

    if sum_metabolite is not None and sum_metapathway is not None and sum_customerlist is not None:
        obsp_names.append(database_name+'-total-total')

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
        S[index_obsp] = adata.obsp['MetaChat-' + obsp_names[index_obsp]]
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
        adata.uns['MetaChat_cluster_spatial-'+clustering+'-'+obsp_names[index_obsp]] = {'communication_matrix': df_cluster, 'communication_pvalue': df_p_value}
    
    return adata if copy else None

################## MCC pathway summary ##################
def summary_pathway(adata: anndata.AnnData,
                    database_name: str = None,
                    sender_cluster: str = None,
                    receiver_cluster: str = None,
                    usenumber_metapathway: int = 5,
                    permutation_spatial: bool = False,
                    clustering: str = None,
                    ):
    df_MetaSen = adata.uns["Metabolite_Sensor_filtered"].copy()
    df_MetaSen['Metapathway'] = df_MetaSen['Metapathway'].str.replace(" / ", "_", regex=False)
    Metapathway_data = df_MetaSen["Metapathway"].copy()
    Metapathway_list = []
    for item in Metapathway_data:
        split_items = item.split('; ')
        Metapathway_list.extend(split_items)
    sum_metapathway = np.unique(Metapathway_list).tolist()
    sum_metapathway = [x for x in sum_metapathway if x != 'nan']

    # Choose the most significant metabolic pathway in the communication between these sender cluster and receiver cluster
    MCC_metapathway = pd.DataFrame(np.zeros((len(sum_metapathway),2)), index=sum_metapathway, columns=['communication_score','p_value'])
    for name_pathway in MCC_metapathway.index:
        if permutation_spatial == True:
            MCC_metapathway.loc[name_pathway,"communication_score"] = adata.uns["MetaChat_cluster_spatial-" + clustering + "-" + database_name + "-" + name_pathway]["communication_matrix"].loc[sender_cluster,receiver_cluster]
            MCC_metapathway.loc[name_pathway,"p_value"] = adata.uns["MetaChat_cluster_spatial-" + clustering + "-" + database_name + "-" + name_pathway]["communication_pvalue"].loc[sender_cluster,receiver_cluster]
        else:
            MCC_metapathway.loc[name_pathway,"communication_score"] = adata.uns["MetaChat_cluster-" + clustering + "-" + database_name + "-" + name_pathway]["communication_matrix"].loc[sender_cluster,receiver_cluster]
            MCC_metapathway.loc[name_pathway,"p_value"] = adata.uns["MetaChat_cluster-" + clustering + + "-" + database_name + "-" + name_pathway]["communication_pvalue"].loc[sender_cluster,receiver_cluster]
      
    metapathway_rank = MCC_metapathway.sort_values(by=['p_value', 'communication_score'], ascending=[True, False])
    use_metapatheway = metapathway_rank.index[:usenumber_metapathway,].tolist()

    # Compute the each m-s pairs communication_score
    MCC_cluster_pair = adata.uns['Metabolite_Sensor_filtered'].copy()
    MCC_cluster_pair['Metapathway'] = MCC_cluster_pair['Metapathway'].str.replace(" / ", "_", regex=False)
    for irow, ele in MCC_cluster_pair.iterrows():
        Metaname = ele['Metabolite']
        Sensname = ele['Sensor']
        MCC_cluster_pair.loc[irow, "communication_score"] = adata.uns["MetaChat_cluster-" + clustering + "-" + database_name + "-" + Metaname + "-" + Sensname]["communication_matrix"].loc[sender_cluster,receiver_cluster]

    MCC_Meta2pathway = MCC_cluster_pair[["Metabolite", "Metapathway", "Sensor", "Sensorpathway", "communication_score"]]
    MCC_Meta2pathway = MCC_Meta2pathway[((MCC_Meta2pathway['Metapathway'] != "nan") & (MCC_Meta2pathway['Sensorpathway'] != "nan"))]
    MCC_Meta2pathway['Metapathway'] = MCC_Meta2pathway['Metapathway'].str.split('; ')
    MCC_Meta2pathway_expanded1 = MCC_Meta2pathway.explode('Metapathway')
    MCC_Meta2pathway_expanded1['Sensorpathway'] = MCC_Meta2pathway_expanded1['Sensorpathway'].str.split('; ')
    MCC_Meta2pathway_expanded2 = MCC_Meta2pathway_expanded1.explode('Sensorpathway')
    MCC_Meta2pathway_group = MCC_Meta2pathway_expanded2.groupby(['Metapathway', 'Sensorpathway'], as_index=False).agg({'communication_score': 'sum'})
    filtered_MCC_Meta2pathway_group = MCC_Meta2pathway_group[MCC_Meta2pathway_group["Metapathway"].isin(use_metapatheway)]

    # construct graph network to measure importance
    G = nx.DiGraph()
    edges_with_weights = [
        (row['Metapathway'], row['Sensorpathway'], row['communication_score']) 
        for _, row in filtered_MCC_Meta2pathway_group.iterrows()
    ]
    for edge in edges_with_weights:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    hubs, authorities = nx.hits(G, max_iter=500, normalized=True)
    senspathway_rank = sorted(authorities.items(), key=lambda item: item[1], reverse=True)
    senspathway_rank = pd.DataFrame(senspathway_rank, columns=['Senspathway', 'Rankscore'])
    senspathway_rank = senspathway_rank[senspathway_rank['Senspathway'].str.startswith('WP')]

    ms_result = filtered_MCC_Meta2pathway_group.pivot_table(index='Metapathway', columns='Sensorpathway', values='communication_score')
    ms_result = ms_result.fillna(0)

    return metapathway_rank, senspathway_rank, ms_result

################## MCC remodelling ##################
def communication_deg_detection(
    adata: anndata.AnnData,
    adata_deg: anndata.AnnData,
    n_var_genes: int = None,
    var_genes = None,
    database_name: str = None,
    sum_metabolite: str = None,
    sum_metapathway: str = None,
    sum_customerlist: str = None,
    summary: str = 'receiver',
    nknots: int = 6,
    n_deg_genes: int = None,
    n_points: int = 50,
    deg_pvalue_cutoff: float = 0.05,
):

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

    adata_deg_raw = adata_deg.copy()
    adata_deg_var = adata_deg.copy()
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

    if sum_metabolite is None and sum_metapathway is None and sum_customerlist is None:
        sum_name = 'total-total'
        obsm_name = ''
    elif sum_metabolite is not None:
        sum_name = sum_metabolite
        obsm_name = '-metabolite'
    elif sum_metapathway is not None:
        sum_name = sum_metapathway
        obsm_name = '-pathway'
    elif sum_customerlist is not None:
        sum_name = sum_customerlist
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
    # df_deg = df_deg.iloc[:,0:3]
    df_yhat = yhat_scaled

    anndata2ri.deactivate()
    ro.numpy2ri.deactivate()
    ro.pandas2ri.deactivate()

    return df_deg, df_yhat
    
def communication_deg_clustering(
    df_deg: pd.DataFrame,
    df_yhat: pd.DataFrame,
    deg_clustering_npc: int = 10,
    deg_clustering_knn: int = 5,
    deg_clustering_res: float = 1.0,
    n_deg_genes: int = 200,
    p_value_cutoff: float = 0.05
):
    """
    Cluster the communcation DE genes based on their fitted expression pattern.

    Parameters
    ----------
    df_deg
        The deg analysis summary data frame obtained by running ``commot.tl.communication_deg_detection``.
        Each row corresponds to one tested genes and columns include "waldStat" (Wald statistics), "df" (degrees of freedom), and "pvalue" (p-value of the Wald statistics).
    df_yhat
        The fitted (smoothed) gene expression pattern obtained by running ``commot.tl.communication_deg_detection``.
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

def communication_deg_keggEnrich(
    gene_list: list = None,
    gene_sets: str = "KEGG_2021_Human",
    organism: str = "Human"
):
    # mouse gene_set 'KEGG_2019_Mouse'
    enr = gp.enrichr(gene_list = gene_list,
                     gene_sets = gene_sets,
                     organism = organism,
                     no_plot = True,
                     cutoff = 0.5)
    df_result = enr.results
    return df_result