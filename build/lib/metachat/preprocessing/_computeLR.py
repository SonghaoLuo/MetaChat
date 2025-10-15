import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import networkx as nx
import matplotlib as mpl
from pydpc import Cluster
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial import distance_matrix

def LRC_unfiltered(adata: anndata.AnnData,
                   LRC_name: str = None,
                   LRC_marker_gene: str = None,  
                   quantile: float = 90.0,
                   copy: bool = False):
    """
    Function for identifying LRC candidate (unfiltered) points based on the quantile of the marker gene.

    Parameters
    ----------
    adata
        The data matrix with shape ``n_obs`` × ``n_var``, provided as an `anndata` object.
        Rows correspond to cells or spots and columns to genes.
    LRC_name
        The name of the long-range channel, provided as a string, such as "Blood" or "CSF" (cerebrospinal fluid).
    LRC_marker_gene
        The name of the marker gene used for identifying `LRC_name`, provided as a string.
    quantile
        The quantile threshold for selecting candidates, provided as a float between 0.0 and 100.0. 
        For example, a value of 90.0 means selecting points greater than the 90th percentile.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.

    Returns
    -------
    adata : anndata.AnnData
        The candidate points are added to ``.obs['LRC_' + LRC_name + '_unfiltered']``, with "1" indicating a candidate and "0" indicating not a candidate. 
        If copy=True, return the AnnData object and return None otherwise.

    """

    assert LRC_name is not None, "Please provide an LRC name."
    assert LRC_marker_gene is not None, f"Please provide a marker gene for the LRC: {LRC_name}."

    # Compute the quantile based on the marker gene expression of the long-range channel.
    threshold = np.percentile(adata[:,LRC_marker_gene].X.toarray(), q=quantile)

    # Identify candidate cells where the expression of the long-range marker gene exceeds the quantile threshold
    candidate_cells = adata[:,LRC_marker_gene].X.toarray().flatten() > threshold

    # Store the candidate cells in the AnnData object.
    candidate_cells_int = candidate_cells.astype(int)
    candidate_cells_cat = pd.Categorical(candidate_cells_int)
    adata.obs['LRC_' + LRC_name + '_unfiltered'] = candidate_cells_cat
    print(f"Cells with {LRC_marker_gene} expression above the {quantile}% have been selected as candidates and stored in 'adata.obs['LRC_{LRC_name}_unfiltered']'.")

    return adata if copy else None

def LRC_cluster(adata: anndata.AnnData, 
                LRC_name: str = None,
                density_cutoff: float = 10.0,
                delta_cutoff: float = 10.0,
                outlier_cutoff: float = 2.0, 
                fraction: float = 0.02,
                plot_savepath: str = None):
    """
    Function for performing local density clustering on candidate points.

    Parameters
    ----------
    adata
        The data matrix with shape ``n_obs`` × ``n_var``, provided as an `anndata` object.
        Rows correspond to cells or spots and columns to genes.
    LRC_name
        The name of the long-range channel, provided as a string, such as "Blood" or "CSF" (cerebrospinal fluid).
    density_cutoff
        Density threshold for screening center points, provided as a float. Points that exceed both the ``density_cutoff`` and ``delta_cutoff`` are selected as center points.
    delta_cutoff
        Density threshold for screening center points, provided as a float. Points that exceed both the ``density_cutoff`` and ``delta_cutoff`` are selected as center points.
    outlier_cutoff
        Threshold for filtering outlier points, provided as a float.
    fraction
        The fraction of points relative to the total number of points to calculate local density and delta, provided as a float.
    plot_savepath
        The save path for the output plot of the clustering algorithm, provided as a string.
        
    Returns
    -------
    LRC_cluster : pydpc.dpc.Cluster
        This is the class object that is the output of the local density clustering algorithm, containing information about when the algorithm was used, and will be used as input to the ``mc.pp.LRC_filtered`` function.
    
    """
    # Check inputs
    assert LRC_name is not None, "Please provide an LRC name."
    key = 'LRC_' + LRC_name + '_unfiltered'
    if not key in adata.obs.keys():
        raise KeyError("Please run the mc.pp.LRC_unfiltered function first")

    # Convert the LRC data to boolean to filter the cells
    LRC_cellsIndex = adata.obs[key].astype(bool)

    # Extract spatial coordinates of the filtered cells
    points = adata[LRC_cellsIndex,:].obsm['spatial'].toarray().astype('double')

    # Create a Cluster object with the spatial points and specified fraction
    LRC_cluster = Cluster(points, fraction, autoplot=False)

    # Disable autoplot for the cluster
    LRC_cluster.autoplot=False

    # Assign clusters based on density and delta cutoffs
    LRC_cluster.assign(density_cutoff, delta_cutoff)

    # Mark outliers in the cluster based on density
    LRC_cluster.outlier = LRC_cluster.border_member
    LRC_cluster.outlier[LRC_cluster.density <= outlier_cutoff] = True
    LRC_cluster.outlier[LRC_cluster.density > outlier_cutoff] = False

    
    fig, ax = plt.subplots(1,2,figsize=(10, 5))
    # Plot density vs. delta in the first subplot
    ax[0].scatter(LRC_cluster.density, LRC_cluster.delta, s=10)
    ax[0].plot([LRC_cluster.min_density, LRC_cluster.density.max()], [LRC_cluster.min_delta, LRC_cluster.min_delta], linewidth=2, color="red")
    ax[0].plot([LRC_cluster.min_density, LRC_cluster.min_density], [LRC_cluster.min_delta,  LRC_cluster.delta.max()], linewidth=2, color="red")
    ax[0].plot([outlier_cutoff, outlier_cutoff], [0,  LRC_cluster.delta.max()], linewidth=2, color="red", linestyle='--')
    ax[0].set_xlabel(r"density")
    ax[0].set_ylabel(r"delta / a.u.")
    ax[0].set_box_aspect(1)
    # Plot the spatial distribution of points in the second subplot
    ax[1].scatter(points[~LRC_cluster.outlier,0], points[~LRC_cluster.outlier,1], s=5, c=LRC_cluster.membership[~LRC_cluster.outlier], cmap=mpl.cm.tab10)
    ax[1].scatter(points[LRC_cluster.outlier,0], points[LRC_cluster.outlier,1], s=5, c="grey")
    ax[1].invert_yaxis()
    ax[1].set_box_aspect(1)

    # Save the plot if a save path is provided, otherwise show the plot
    if plot_savepath is not None:
        plt.savefig(plot_savepath)
    else:
        plt.show()

    # Return the cluster object
    return LRC_cluster

def LRC_filtered(adata: anndata.AnnData, 
                 LRC_name: str = None,
                 LRC_cluster = None,
                 copy: bool = False):
    """
    Function for assigning LRC points after locak density clustering.

    Parameters
    ----------
    adata
        The data matrix with shape ``n_obs`` × ``n_var``, provided as an `anndata` object.
        Rows correspond to cells or spots and columns to genes.
    LRC_name
        The name of all possible long-range channel, provided as a string, such as "Blood" or "CSF" (cerebrospinal fluid).
    LRC_cluster
        The output of ``mc.pp.LRC_cluster`` funciton
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        The points which belongs to LRC are added to ``.obs['LRC_' + LRC_name + '_filtered']``, with non-"0" indicating the cluster of LRC and "0" indicating not a LRC. 
        If copy=True, return the AnnData object and return None otherwise.    
    """
    
    # Check inputs
    assert LRC_name is not None, "Please provide an LRC name."
    assert LRC_cluster is not None, "Please provide LRC_cluster."
    key = 'LRC_' + LRC_name + '_unfiltered'
    if not key in adata.obs.keys():
        raise KeyError("Please run the 'mc.pp.LRC_unfiltered' function and 'mc.pp.LRC_cluster' function first")

    # Increment the LRC_cluster membership by 1, so that cluster numbering starts from 1 instead of 0
    newcluster = LRC_cluster.membership + 1

    # Set the newcluster values of points marked as outliers to 0
    newcluster[LRC_cluster.outlier] = 0

    # Convert the unfiltered LRC data to integers and store it in a new obs column, then replace the values equal to 1 with the values from newcluster and convert the column to a categorical type
    adata.obs['LRC_' + LRC_name + '_filtered'] = adata.obs['LRC_' + LRC_name + '_unfiltered'].astype(int)
    adata.obs['LRC_' + LRC_name + '_filtered'][adata.obs['LRC_' + LRC_name + '_filtered'] == 1] = newcluster
    adata.obs['LRC_' + LRC_name + '_filtered'] = adata.obs['LRC_' + LRC_name + '_filtered'].astype('category')

    print(f"Candidate points for {LRC_name} LRC are clustered and outliers are removed. LRC points are stored in 'adata.obs['LRC_{LRC_name}_filtered']'.")

    return adata if copy else None

def compute_costDistance(adata: anndata.AnnData,
                         LRC_type: list = None,
                         dis_thr: float = 50.0,
                         k_neighb: int = 5,
                         LRC_strength: float = 100.0,
                         plot: bool = False,
                         spot_size: int = None,
                         copy: bool = False):
    """
    Function for computing LRC-embedding cost distance based on graph construction.

    Parameters
    ----------
    adata
        The data matrix with shape ``n_obs`` × ``n_var``, provided as an `anndata` object.
        Rows correspond to cells or spots and columns to genes.
    LRC_type
        The name of all possible long-range channel, provided as a `list`, such as ["Blood"] or ["CSF"] or ["Blood","CSF"].
    dis_thr
        The farthest distance from a nearby cell to the LRC indicates the range of cells in which long-range communication can occur, provided as a `float`.
    k_neighb
        k-nearest neighbors for constructing graph, provided as an `int`.
    LRC_strength
        Long-range communication strength for computing LRC-embedding cost distance, provided as a `float`.
    plot   
        Whether or not to plot diagrams showing cells where long-range communication may occur, for each type of LRC and each LRCs in a type. 
    spot_size
        If plot=True, The size of the spot in the figure. When .uns['spatial'][library_id] does not exist, spot_size must be provided directly.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        The calculated distance will be saved in ``adata.obsp['spatial_distance_LRC_No']`` and ``adata.obsp['spatial_distance_LRC_X']`` for each LRC, where 'X' is the LRC name.
        If copy=True, return the AnnData object and return None otherwise.    
    """
    
    # Check inputs
    if LRC_type is None: 
        print("You didn't input LRC_type, so long-range communication will not be consider in subsequence analysis")
    
    print("Compute spatial cost distance without long-range channel...")
    if not 'spatial_distance' in adata.obsp.keys():
        dis_mat = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])
        adata.obsp['spatial_distance_LRC_No'] = dis_mat
    else:
        dis_mat = adata.obsp['spatial_distance']
        adata.obsp['spatial_distance_LRC_No'] = dis_mat

    # Compute spatial cost distance incorporating long-range channel
    if LRC_type is not None:
        for LRC_element in LRC_type:

            # Check inputs
            key = 'LRC_' + LRC_element + '_filtered'
            if not key in adata.obs.keys():
                raise KeyError(f"Can't find the adata.obs[{key}], Please run the 'mc.pp.LRC_unfiltered' function, 'mc.pp.LRC_cluster' function and 'mc.pp.LRC_filtered' function first")
            
            print("Compute spatial cost distance incorporating long-range channel of " + LRC_element)
            spot_close_LR = []
            spot_close_LR_type = []
            LR_set = set(adata.obs['LRC_' + LRC_element + '_filtered'])
            LR_set.remove(0)

            # Find out the points that close to LRC
            record_closepoint = np.zeros((len(adata.obs['LRC_' + LRC_element + '_filtered']), len(LR_set)))
            for ispot in range(dis_mat.shape[0]):
                spot_close_ind = dis_mat[ispot,:] < dis_thr
                temp_spot_close = adata.obs['LRC_' + LRC_element + '_filtered'][spot_close_ind]
                if np.any(temp_spot_close != 0):
                    spot_close_LR.append(ispot)   
                    LR_type = set(temp_spot_close[temp_spot_close!=0])
                    record_closepoint[ispot,np.array(list(LR_type),dtype=int)-1] = 1
                    spot_close_LR_type.append(LR_type)
            spot_close_LR = np.array(spot_close_LR)

            # plot close points
            for itype in LR_set:
                adata.obs['LRC_' + LRC_element + '_closepoint_cluster%d' %itype] = record_closepoint[:,int(itype-1)]
                if plot:
                    print('The points that can occur long-range communication is...')
                    if spot_size is not None:
                        sc.pl.spatial(adata, color='LRC_' + LRC_element + '_closepoint_cluster%d' %itype, spot_size=spot_size)
                    else:
                        sc.pl.spatial(adata, color='LRC_' + LRC_element + '_closepoint_cluster%d' %itype)
                else: 
                    pass 

            # Compute the distance between two arbitary point for each long-range channel
            LR_set = set(adata.obs['LRC_' + LRC_element + '_filtered'])
            LR_set.remove(0)
            G_list = []
            
            print("  Construct network graph of long-range channel among %d neighborhoods..." %k_neighb)
            for itype in LR_set:
                itype = int(itype)
                G = nx.Graph()
                LR_channel_coords = adata.obsm['spatial'][adata.obs['LRC_' + LRC_element + '_filtered'] == itype]

                # Add nodes
                for iLR in range(LR_channel_coords.shape[0]):
                    x_coord, y_coord = LR_channel_coords[iLR]
                    G.add_node((x_coord, y_coord))

                # Add edges between k_neighb neighborh
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
                
                LR_channel_coords = adata.obsm['spatial'][adata.obs['LRC_' + LRC_element + '_filtered'] == itype]
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
            LR_set = set(adata.obs['LRC_' + LRC_element + '_filtered'])
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
                            spot_itype_ind = adata.obs['LRC_' + LRC_element + '_filtered'] == itype
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
            adata.obsp['spatial_distance_LRC_' + LRC_element] = dis_mat_LR_min
    print("Finished!")

    return adata if copy else None