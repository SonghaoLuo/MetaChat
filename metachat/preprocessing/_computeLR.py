import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import networkx as nx
import matplotlib as mpl
from pydpc import Cluster
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from itertools import combinations

def LRC_unfiltered(adata: anndata.AnnData,
                   LRC_name: str,
                   LRC_marker_gene: str,  
                   quantile: float = 90.0):

    # Compute the 90% quantile of the long-range marker gene expression
    threshold_mg = np.percentile(adata[:, LRC_marker_gene].X.toarray(), q=quantile)
    # Filter cells based on long-range marker gene expression exceeding the quantile threshold
    filtered_cells = adata[:, LRC_marker_gene].X.toarray() > threshold_mg
    adata.obs['LRC_' + LRC_name + '_unfiltered'] = filtered_cells.astype(float)


def LRC_cluster(adata: anndata.AnnData, 
                LRC_name: str,
                density: float,
                delta: float,
                outlier_cutoff: float = 2, 
                save: bool = False, 
                fraction: float = 0.02):

    key = 'LRC_' + LRC_name + '_unfiltered'
    if not key in adata.obs.keys():
        raise KeyError("Please run the LRC_unfiltered function first")

    LRC_cellsIndex = adata.obs[key].astype(bool)
    points = adata[LRC_cellsIndex,:].obsm['spatial'].toarray().astype('double')
    LRC_cluster = Cluster(points, fraction, autoplot=False)
    LRC_cluster.autoplot=False
    LRC_cluster.assign(density, delta)

    LRC_cluster.outlier = LRC_cluster.border_member
    LRC_cluster.outlier[LRC_cluster.density <= outlier_cutoff] = True
    LRC_cluster.outlier[LRC_cluster.density > outlier_cutoff] = False
  
    fig, ax = plt.subplots(1,2,figsize=(10, 5))
    ax[0].scatter(LRC_cluster.density, LRC_cluster.delta, s=10)
    ax[0].plot([LRC_cluster.min_density, LRC_cluster.density.max()], [LRC_cluster.min_delta, LRC_cluster.min_delta], linewidth=2, color="red")
    ax[0].plot([LRC_cluster.min_density, LRC_cluster.min_density], [LRC_cluster.min_delta,  LRC_cluster.delta.max()], linewidth=2, color="red")
    ax[0].plot([outlier_cutoff, outlier_cutoff], [0,  LRC_cluster.delta.max()], linewidth=2, color="red", linestyle='--')
    ax[0].set_xlabel(r"density")
    ax[0].set_ylabel(r"delta / a.u.")
    ax[0].set_box_aspect(1)
    ax[1].scatter(points[~LRC_cluster.outlier,0], points[~LRC_cluster.outlier,1], s=2, c=LRC_cluster.membership[~LRC_cluster.outlier], cmap=mpl.cm.tab10)
    ax[1].scatter(points[LRC_cluster.outlier,0], points[LRC_cluster.outlier,1], s=2, c="grey")
    ax[1].invert_yaxis()
    ax[1].set_box_aspect(1)

    if save:
        plt.savefig(save)
    else:
        plt.show()

    return LRC_cluster

def LRC_filtered(adata: anndata.AnnData, 
                 LRC_name: str,
                 LRC_cluster):

    key = 'LRC_' + LRC_name + '_unfiltered'
    if not key in adata.obs.keys():
        raise KeyError("Please run the 'LRC_unfiltered' function and 'LRC_cluster' function first")

    newcluster = LRC_cluster.membership + 1
    newcluster[LRC_cluster.outlier] = 0
    adata.obs['LRC_' + LRC_name + '_filtered'] = adata.obs['LRC_' + LRC_name + '_unfiltered']
    adata.obs['LRC_' + LRC_name + '_filtered'][adata.obs['LRC_' + LRC_name + '_filtered'] == 1] = newcluster
    adata.obs['LRC_' + LRC_name + '_filtered'] = adata.obs['LRC_' + LRC_name + '_filtered'].astype('category')

    return

def compute_longRangeDistance(adata: anndata.AnnData,
                              database_name: str = None,
                              df_MetaSen: pd.DataFrame = None,
                              LRC_name: list = None,
                              dis_thr: float = 50,
                              k_neighb: int = 5,
                              LRC_strength: float = 100,
                              plot = True,
                              spot_size = None
                              ):
    

    ### Check database ### 
    assert database_name is not None, "Please give a database_name"
    assert df_MetaSen is not None, "Please give a Metabolite-Sensor database"

    print("Compute spatial distance without long-range channel...")
    if not 'spatial_distance' in adata.obsp.keys():
        dis_mat = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])
        adata.obsp['spatial_distance_LRC_No'] = dis_mat
    else:
        dis_mat = adata.obsp['spatial_distance']
        adata.obsp['spatial_distance_LRC_No'] = dis_mat

    # remove unavailable genes from df_MetaSen
    data_var = list(adata.var_names)
    tmp_MetaSen = []
    for i in range(df_MetaSen.shape[0]):
        if df_MetaSen.loc[i,"Metabolite"] in data_var and df_MetaSen.loc[i,"Sensor"] in data_var:
            tmp_MetaSen.append(df_MetaSen.loc[i,:])
    tmp_MetaSen = np.array(tmp_MetaSen, str)
    df_MetaSen_filtered = pd.DataFrame(data = tmp_MetaSen)
    df_MetaSen_filtered.columns = df_MetaSen.columns

    # Drop duplicate pairs
    df_MetaSen_filtered = df_MetaSen_filtered.drop_duplicates()
    adata.uns["Metabolite_Sensor_filtered"] = df_MetaSen_filtered
    print("There are %d pairs were found from the spatial data." %df_MetaSen_filtered.shape[0])

    ### Compute new spatial distance incorporating long-range channel ###
    for LR_element in LRC_name:
        print("Compute new spatial distance incorporating long-range channel of " + LR_element)
        spot_close_LR = []
        spot_close_LR_type = []

        LR_set = set(adata.obs['LRC_' + LR_element + '_filtered'])
        LR_set.remove(0)

        record_closepoint = np.zeros((len(adata.obs['LRC_' + LR_element + '_filtered']), len(LR_set)))

        for ispot in range(dis_mat.shape[0]):
            spot_close_ind = dis_mat[ispot,:] < dis_thr
            temp_spot_close = adata.obs['LRC_' + LR_element + '_filtered'][spot_close_ind]
            if np.any(temp_spot_close != 0):
                spot_close_LR.append(ispot)   
                LR_type = set(temp_spot_close[temp_spot_close!=0])
                record_closepoint[ispot,np.array(list(LR_type),dtype=int)-1] = 1
                spot_close_LR_type.append(LR_type)
        spot_close_LR = np.array(spot_close_LR)

        # plot close points
        for itype in LR_set:
            adata.obs['LRC_' + LR_element + '_closepoint_cluster%d' %itype] = record_closepoint[:,int(itype-1)]
            if plot:
                if spot_size is not None:
                    sc.pl.spatial(adata, color='LRC_' + LR_element + '_closepoint_cluster%d' %itype, spot_size=spot_size)
                else:
                    sc.pl.spatial(adata, color='LRC_' + LR_element + '_closepoint_cluster%d' %itype)
            else: 
                pass 

        # Compute the distance between two arbitary point for each long-range channel
        LR_set = set(adata.obs['LRC_' + LR_element + '_filtered'])
        LR_set.remove(0)
        G_list = []
        
        print("  Construct network graph of long-range channel among %d neighborhoods..." %k_neighb)
        for itype in LR_set:
            itype = int(itype)
            G = nx.Graph()
            LR_channel_coords = adata.obsm['spatial'][adata.obs['LRC_' + LR_element + '_filtered'] == itype]

            # add nodes
            for iLR in range(LR_channel_coords.shape[0]):
                x_coord, y_coord = LR_channel_coords[iLR]
                G.add_node((x_coord, y_coord))

            # add edges between 5 neighborh
            dis_mat_LR_point = distance_matrix(LR_channel_coords, LR_channel_coords)
        
            for iLR in range(dis_mat_LR_point.shape[0]):
                x_coord, y_coord = LR_channel_coords[iLR]
                min_ind = np.argsort(dis_mat_LR_point[iLR,:])[1:k_neighb+1]
                for ineigh in min_ind:
                    x_coord_neigh, y_coord_neigh = LR_channel_coords[ineigh]
                    G.add_edge((x_coord, y_coord), (x_coord_neigh, y_coord_neigh), weight = dis_mat_LR_point[iLR,ineigh])
            
            if nx.is_connected(G) == False:
                nx.is_connected(G)
                components = list(nx.connected_components(G))
                comb = list(combinations(range(len(components)), 2))
                for (i,j) in comb:
                    components_A = [[x, y] for x, y in components[i]]
                    components_B = [[x, y] for x, y in components[j]]
                    dis_mat_subgroup = distance_matrix(components_A, components_B)
                    min_index_A, min_index_B = np.unravel_index(np.argmin(dis_mat_subgroup), dis_mat_subgroup.shape)
                    x_coord, y_coord = components_A[min_index_A]
                    x_coord_neigh, y_coord_neigh = components_B[min_index_B]
                    G.add_edge((x_coord, y_coord), (x_coord_neigh, y_coord_neigh), weight = np.min(dis_mat_subgroup))
                    
            G_list.append(G)

        # Calculate the shortest path distance from the source to the target using the shortest path algorithm
        print("  Calculate the shortest path distance from the source to the target using the shortest path algorithm...")
        dis_LR_path_list = []

        for itype in LR_set:
            itype = int(itype)
            dis_LR_path = {}
            
            LR_channel_coords = adata.obsm['spatial'][adata.obs['LRC_' + LR_element + '_filtered'] == itype]
            G = G_list[itype-1]
            
            print("    For the long-range case of cluster %s..." %itype)
            for m_ind in tqdm(range(len(LR_channel_coords))):
                for n_ind in range(len(LR_channel_coords)):
                    x_coord_m, y_coord_m = LR_channel_coords[m_ind]
                    x_coord_n, y_coord_n = LR_channel_coords[n_ind]
                    precision_m = len(str(x_coord_m).split('.')[-1])
                    precision_n = len(str(x_coord_n).split('.')[-1])
                    pathname = "source({:.{}f},{:.{}f})-target({:.{}f},{:.{}f})".format(x_coord_m, precision_m, y_coord_m, precision_m, x_coord_n, precision_n, y_coord_n, precision_n)
                    dis_LR_path[pathname] = nx.dijkstra_path_length(G, source = (x_coord_m, y_coord_m), target = (x_coord_n, y_coord_n), weight = 'weight')
                    
            dis_LR_path_list.append(dis_LR_path)
        
        print("  Rearrange distance matrix...")
        LR_set = set(adata.obs['LRC_' + LR_element + '_filtered'])
        dis_mat_LR = np.zeros((len(LR_set), dis_mat.shape[0], dis_mat.shape[0]))

        for itype in LR_set:
            
            itype = int(itype)

            ## problem！！！##
            dis_LR_path = dis_LR_path_list[itype-1].copy()
            
            if itype == 0:
                dis_mat_LR[itype,:,:] = dis_mat.copy()
            else:
                print("    For the long-range case of cluster %s..." %itype)
                spot_close_LR_itype = spot_close_LR[np.where(np.array([itype in set_obj for set_obj in spot_close_LR_type]) == True)]
                dis2LR = []
                closest_spot = []
                for ispot in spot_close_LR_itype:
                        spot_close_ind = dis_mat[ispot,:] < dis_thr
                        spot_itype_ind = adata.obs['LRC_' + LR_element + '_filtered'] == itype
                        dis2LR_temp = np.min(dis_mat[ispot,spot_close_ind & spot_itype_ind])
                        dis2LR.append(dis2LR_temp)

                        closest_temp = np.argmin(dis_mat[ispot,spot_close_ind & spot_itype_ind])
                        closest_ind = np.where(spot_close_ind & spot_itype_ind)[0][closest_temp]
                        closest_spot_temp = adata.obsm['spatial'][closest_ind]
                        closest_x, closest_y = closest_spot_temp
                        closest_spot.append((closest_x,closest_y))
                dis2LR = np.array(dis2LR)        
                dis_LR = np.tile(dis2LR, (len(dis2LR),1)) + np.tile(dis2LR, (len(dis2LR),1)).T
                
                dis_mat_LR_path = np.zeros((len(closest_spot),len(closest_spot)))

                for m_ind in tqdm(range(dis_mat_LR_path.shape[0])):
                    for n_ind in range(dis_mat_LR_path.shape[1]):
                        x_coord_m, y_coord_m = closest_spot[m_ind]
                        x_coord_n, y_coord_n = closest_spot[n_ind]
                        precision_m = len(str(x_coord_m).split('.')[-1])
                        precision_n = len(str(x_coord_n).split('.')[-1])
                        pathname = "source({:.{}f},{:.{}f})-target({:.{}f},{:.{}f})".format(x_coord_m, precision_m, y_coord_m, precision_m, x_coord_n, precision_n, y_coord_n, precision_n)
                        dis_mat_LR_path[m_ind,n_ind] = dis_LR_path[pathname]

                dis_mat_temp = dis_mat.copy()
                dis_mat_temp = pd.DataFrame(dis_mat_temp)
                dis_mat_temp.iloc[spot_close_LR_itype,spot_close_LR_itype] = dis_LR + dis_mat_LR_path/LRC_strength
                dis_mat_LR[itype,:,:] = np.array(dis_mat_temp)
        dis_mat_LR_min = np.min(dis_mat_LR, axis=0)
        adata.obsp['spatial_distance_LRC_' + LR_element] = dis_mat_LR_min
    print("Finished!")
