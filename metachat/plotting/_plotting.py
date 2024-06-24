from typing import Optional
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pycirclize import Circos
import networkx as nx

from .._utils import plot_cell_signaling
from .._utils import get_cmap_qualitative

def plot_communication_flow(
    adata: anndata.AnnData,
    database_name: str = None,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    plot_method: str = "cell",
    background: str = "summary",
    background_legend: bool=False,
    library_id: str = None,
    clustering: str = None,
    summary: str = "sender",
    cmap: str = "coolwarm",
    cluster_cmap: dict = None,
    pos_idx: np.ndarray = np.array([0,1],int),
    ndsize: float = 1,
    scale: float = 1.0,
    normalize_summary_quantile: float = 0.995,
    normalize_v: bool = False,
    normalize_v_quantile: float = 0.95,
    arrow_color: str = "#333333",
    grid_density: float = 1.0,
    grid_knn: int = None,
    grid_scale: float = 1.0,
    grid_thresh: float = 1.0,
    grid_width: float = 0.005,
    stream_density: float = 1.0,
    stream_linewidth: float = 1,
    stream_cutoff_perc: float = 5,
    title: str = None,
    filename: str = None,
    ax: Optional[mpl.axes.Axes] = None
):
    """
    Plot spatial directions of cell-cell communication.
    
    .. image:: cell_communication.png
        :width: 500pt

    The cell-cell communication should have been computed by the function :func:`commot.tl.spatial_communication`.
    The cell-cell communication direction should have been computed by the function :func:`commot.tl.communication_direction`.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the ligand-receptor interaction database. 
    pathway_name
        Name of the signaling pathway to be plotted. Will be used only when ``lr_pair`` and ``keys`` are None.
        If none of ``pathway_name``, ``lr_pair``, and ``keys`` are given, the total signaling through all pairs will be plotted.
    lr_pair
        A tuple of ligand name and receptor name. If given, ``pathway_name`` will be ignored. Will be ignored if ``keys`` is given.
    keys
        A list of keys for example 'ligA-recA' for a LR pair or 'pathwayX' for a signaling pathway 'pathwayX'. 
        If given, pathway_name and lr_pair will be ignored.
        If more than one is given, the average will be plotted.
    plot_method
        'cell' plot vectors on individual cells. 
        'grid' plot interpolated vectors on regular grids.
        'stream' streamline plot.
    background
        'summary': scatter plot with color representing total sent or received signal.
        'image': the image in Visium data.
        'cluster': scatter plot with color representing cell clusters.
    background_legend
        Whether to include the background legend when background is set to `summary` or `cluster`.
    clustering
        The key for clustering result. Needed if background is set to `cluster`.
        For example, if ``clustering=='leiden'``, the clustering result should be available in ``.obs['leiden']``.
    summary
        If background is set to 'summary', the numerical value to plot for background.
        'sender': node color represents sender weight.
        'receiver': node color represents receiver weight.
    cmap
        matplotlib colormap name for node summary if numerical (background set to 'summary'), e.g., 'coolwarm'.
        plotly colormap name for node color if summary is (background set to 'cluster'). e.g., 'Alphabet'.
    cluster_cmap
        A dictionary that maps cluster names to colors when setting background to 'cluster'. If given, ``cmap`` will be ignored.
    pos_idx
        The coordinates to use for plotting (2D plot).
    ndsize
        The node size of the spots.
    scale
        The scale parameter passed to the matplotlib quiver function :func:`matplotlib.pyplot.quiver` for vector field plots.
        The smaller the value, the longer the arrows.
    normalize_v
        Whether the normalize the vector field to uniform lengths to highlight the directions without showing magnitudes.
    normalize_v_quantile
        The vector length quantile to use to normalize the vector field.
    arrow_color
        The color of the arrows.
    grid_density
        The density of grid if ``plot_method=='grid'``.
    grid_knn
        If ``plot_method=='grid'``, the number of nearest neighbors to interpolate the signaling directions from spots to grid points.
    grid_scale
        The scale parameter (relative to grid size) for the kernel function of mapping directions of spots to grid points.
    grid_thresh
        The threshold of interpolation weights determining whether to include a grid point. A smaller value gives a tighter cover of the tissue by the grid points.
    grid_width
        The value passed to the ``width`` parameter of the function :func:`matplotlib.pyplot.quiver` when ``plot_method=='grid'``.
    stream_density
        The density of stream lines passed to the ``density`` parameter of the function :func:`matplotlib.pyplot.streamplot` when ``plot_method=='stream'``.
    stream_linewidth
        The width of stream lines passed to the ``linewidth`` parameter of the function :func:`matplotlib.pyplot.streamplot` when ``plot_method=='stream'``.
    stream_cutoff_perc
        The quantile cutoff to ignore the weak vectors. Default to 5 that the vectors shorter than the 5% quantile will not be plotted.
    filename
        If given, save to the filename. For example 'ccc_direction.pdf'.
    ax
        An existing matplotlib ax (`matplotlib.axis.Axis`).

    Returns
    -------
    ax : matplotlib.axis.Axis
        The matplotlib ax object of the plot.

    """

    not_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name])
    assert not_none_count < 2, "Please don't not enter all three tasks (metabolite_name, sum_metapathway, sum_customerlist) at the same time."

    if summary == 'sender':
        summary_abbr = 's'
    elif summary == 'receiver':
        summary_abbr = 'r'

    if metabolite_name is None and metapathway_name is None and customerlist_name is None:
        vf_name = 'total-total'
        sum_name = 'total-total'
        obsm_name = ''
    elif metabolite_name is not None:
        vf_name = metabolite_name
        sum_name = metabolite_name
        obsm_name = '-metabolite'
    elif metapathway_name is not None:
        vf_name = metapathway_name
        sum_name = metapathway_name
        obsm_name = '-pathway'
    elif customerlist_name is not None:
        vf_name = customerlist_name
        sum_name = customerlist_name
        obsm_name = '-customer'

    V = adata.obsm['MetaChat' + '_' + summary + '_vf-' + database_name + '-' + vf_name][:,pos_idx].copy()
    signal_sum = adata.obsm['MetaChat-' + database_name + "-sum-" + summary + obsm_name][summary_abbr + '-' + sum_name].copy()

    if background=='cluster' and not cmap in ['Plotly','Light24','Dark24','Alphabet']:
        cmap='Alphabet'
    if ax is None:
        fig, ax = plt.subplots()
    if normalize_v:
        V = V / np.quantile(np.linalg.norm(V, axis=1), normalize_v_quantile)
    if cluster_cmap is None:
        cluster_cmap = dict(zip(adata.obs[clustering].cat.categories.tolist(), adata.uns[clustering + '_colors']))

    plot_cell_signaling(
        adata.obsm["spatial"][:,pos_idx],
        V,
        signal_sum,
        cmap = cmap,
        cluster_cmap = cluster_cmap,
        arrow_color = arrow_color,
        plot_method = plot_method,
        background = background,
        clustering = clustering,
        background_legend = background_legend,
        library_id = library_id,
        adata = adata,
        summary = summary,
        normalize_summary_quantile = normalize_summary_quantile,
        ndsize = ndsize,
        scale = scale,       
        grid_density = grid_density,
        grid_knn = grid_knn,
        grid_scale = grid_scale,
        grid_thresh = grid_thresh,
        grid_width = grid_width,
        stream_density = stream_density,
        stream_linewidth = stream_linewidth,
        stream_cutoff_perc = stream_cutoff_perc,
        title = title,
        filename = filename,
        ax = ax
    )
    return ax


def plot_communication_dependent_genes(
    df_deg: pd.DataFrame,
    df_yhat: pd.DataFrame,
    show_gene_names: bool = True,
    top_ngene_per_cluster: int = -1,
    colormap: str = 'magma',
    cluster_colormap: str = 'Plotly',
    color_range: tuple = None,
    font_scale: float = 1,
    figsize = (10,10),
    filename = None,
    return_genes = False
):
    """
    Plot smoothed gene expression of the detected communication-dependent genes.
    Takes input from the function :func:`commot.tl.communication_deg_clustering`.

    .. image:: communication_deg.png
        :width: 500pt

    Parameters
    ----------
    df_deg
        A data frame where each row is a gene and 
        the columns should include 'waldStat', 'pvalue', 'cluster'.
        Output of ``tl.communication_deg_clustering``
    df_yhat
        A data frame where each row is the smoothed expression of a gene.
        Output of ``tl.communication_deg_clustering``.
    show_gene_names
        Whether to plot the gene names.
    top_ngene_per_cluster
        If non-negative, plot the top_ngene_per_cluster genes 
        with highest wald statistics.
    colormap
        The colormap for the heatmap. Choose from available colormaps from ``seaborn``.
    cluster_colormap
        The qualitative colormap for annotating gene cluster labels.
        Choose from 'Plotly', 'Alphabet', 'Light24', 'Dark24'.
    font_scale
        Font size.
    filename
        Filename for saving the figure. Set the name to end with '.pdf' or 'png'
        to specify format.
    return_genes
        Whether to return the list of plotted genes.
    
    Returns
    -------
    genes
        Returns the gene list being plotted if return_genes is True.
    """
    cmap = get_cmap_qualitative(cluster_colormap)
    wald_stats = df_deg['waldStat'].values
    labels = np.array( df_deg['cluster'].values, int)
    nlabel = np.max(labels)+1
    yhat_mat = df_yhat.values

    if color_range is not None:
        yhat_mat[yhat_mat > color_range[1]] = color_range[1]
        yhat_mat[yhat_mat < color_range[0]] = color_range[0]

    peak_locs = []
    for i in range(nlabel):
        tmp_idx = np.where(labels==i)[0]
        tmp_y = yhat_mat[tmp_idx,:]
        peak_locs.append(np.mean(np.argmax(tmp_y, axis=1)))
    cluster_order = np.argsort(peak_locs)
    idx = np.array([])
    col_colors = []
    for i in cluster_order:
        tmp_idx = np.where(labels==i)[0]
        tmp_order = np.argsort(-wald_stats[tmp_idx])
        if top_ngene_per_cluster >= 0:
            top_ngene = min(len(tmp_idx), top_ngene_per_cluster)
        else:
            top_ngene = len(tmp_idx)
        idx = np.concatenate((idx, tmp_idx[tmp_order][:top_ngene]))
        for j in range(top_ngene):
            col_colors.append(cmap[i % len(cmap)])

    sns.set(font_scale=font_scale)
    g = sns.clustermap(df_yhat.iloc[idx].T, 
        row_cluster=False, 
        col_cluster=False, 
        col_colors=col_colors,
        cmap = colormap,
        xticklabels = show_gene_names,
        yticklabels = False,
        linewidths=0,
        figsize=figsize)
    g.ax_heatmap.invert_yaxis()
    g.cax.set_position([.1, .2, .03, .45])
    plt.savefig(filename, dpi=300)

    if return_genes:
        return list( df_deg.iloc[idx].index )
    
def plot_cluster_communication_chord(
    adata: anndata.AnnData,
    database_name: str = None,
    clustering: str = None,
    sum_metabolite: str = None,
    sum_metapathway: str = None,
    sum_customerlist: str = None,
    permutation_spatial: bool = False,
    p_value_cutoff: float = 0.05,
    self_communication_off: bool = True,
    highlight_cluster_sender: str = None,
    highlight_cluster_receiver: str = None,
    space: int = 5,
    cmap: str = None,
    save: str = None
    ):

    not_none_count = sum(x is not None for x in [sum_metabolite, sum_metapathway, sum_customerlist])
    assert not_none_count < 2, "Please don't not enter all three tasks (sum_metabolite, sum_metapathway, sum_customerlist) at the same time."

    if sum_metabolite is None and sum_metapathway is None and sum_customerlist is None:
        uns_names = 'total-total'
    elif sum_metabolite is not None:
        uns_names = sum_metabolite
    elif sum_metapathway is not None:
        uns_names = sum_metapathway
    elif sum_customerlist is not None:
        uns_names = sum_customerlist

    if permutation_spatial == True:
        df_communMatrix = adata.uns["MetaChat_cluster_spatial-"  + clustering + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_cluster_spatial-" + clustering + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()
    else:
        df_communMatrix = adata.uns["MetaChat_cluster-" + clustering + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_cluster-" + clustering + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()

    df_communMatrix[df_pvalue > p_value_cutoff] = 0

    if self_communication_off:
        for i in range(df_communMatrix.shape[0]):
            df_communMatrix.iloc[i,i] = 0

    df_communMatrix = df_communMatrix.loc[df_communMatrix.sum(axis=1) != 0]
    df_communMatrix = df_communMatrix.loc[:, df_communMatrix.sum(axis=0) != 0]

    link_kws_handler = None
    if (not highlight_cluster_sender is None) or (not highlight_cluster_receiver is None):
        def link_kws_handler(from_label: str,
                            to_label: str):
            if (not highlight_cluster_sender is None) and (highlight_cluster_receiver is None):
                if from_label in highlight_cluster_sender:
                    return dict(alpha=0.7, zorder=1.0)
                else:
                    return dict(alpha=0.2, zorder=0)
            elif (highlight_cluster_sender is None) and (not highlight_cluster_receiver is None):
                if to_label in highlight_cluster_receiver:
                    return dict(alpha=0.7, zorder=1.0)
                else:
                    return dict(alpha=0.2, zorder=0)
            else:
                if from_label in highlight_cluster_sender or to_label in highlight_cluster_receiver:
                    return dict(alpha=0.7, zorder=1.0)
                else:
                    return dict(alpha=0.2, zorder=0)
    if cmap is None:
        cmap = dict(zip(adata.obs[clustering].cat.categories.tolist(), adata.uns[clustering + '_colors']))      
    if np.sum(np.sum(df_communMatrix)) != 0:
        circos = Circos.initialize_from_matrix(
            df_communMatrix,
            space = space,
            cmap = cmap,
            label_kws = dict(size=12),
            link_kws = dict(ec="black", lw=0.5, direction=1),
            link_kws_handler = link_kws_handler
            )   
        circos.savefig(save)
    else:
        print("There is no significant cluster communication in " + uns_names)


def plot_cluster_communication_heatmap(
    adata: anndata.AnnData,
    database_name: str = None,
    clustering: str = None,
    sum_metabolite: str = None,
    sum_metapathway: str = None,
    sum_customerlist: str = None,
    permutation_spatial: bool = False,
    p_value_plot: bool = True,
    p_value_cutoff: float = 0.05,
    size_scale: int = 300,
    cmap: str = "green",
    marker: str = 's',
    palette = None,
    x_order = None, 
    y_order = None,
    figsize = None,
    ax: Optional[mpl.axes.Axes] = None,
    filename: str = None,
    ):

    not_none_count = sum(x is not None for x in [sum_metabolite, sum_metapathway, sum_customerlist])
    assert not_none_count < 2, "Please don't not enter all three tasks (sum_metabolite, sum_metapathway, sum_customerlist) at the same time."

    if sum_metabolite is None and sum_metapathway is None and sum_customerlist is None:
        uns_names = 'total-total'
    elif sum_metabolite is not None:
        uns_names = sum_metabolite
    elif sum_metapathway is not None:
        uns_names = sum_metapathway
    elif sum_customerlist is not None:
        uns_names = sum_customerlist

    if permutation_spatial == True:
        df_communMatrix = adata.uns["MetaChat_cluster_spatial-"  + clustering + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_cluster_spatial-" + clustering + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()
    else:
        df_communMatrix = adata.uns["MetaChat_cluster-" + clustering + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_cluster-" + clustering + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()

    df_communMatrix = df_communMatrix.reset_index()
    melt_communMatrix = pd.melt(df_communMatrix, id_vars='index', var_name='Column', value_name='Value')
    melt_communMatrix.columns = ['Sender','Receiver','MCC_score']

    df_pvalue = df_pvalue.reset_index()
    melt_pvalue = pd.melt(df_pvalue, id_vars='index', var_name='Column', value_name='Value')
    melt_pvalue.columns = ['Sender','Receiver','p_value']

    melt_df  =pd.concat([melt_communMatrix, melt_pvalue['p_value']], axis=1)

    sender = melt_df['Sender']
    receiver = melt_df['Receiver']
    color = melt_df['MCC_score']
    size = melt_df['MCC_score']
    p_value = melt_df['p_value']

    if palette is not None:
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        if cmap == 'green':
            palette = sns.blend_palette(["#D8E6E5", "#94C1BE", "#49A59D"], n_colors=n_colors)
        elif cmap == 'red':
            palette = sns.blend_palette(["#FCF5B8", "#EBA55A", "#C23532"], n_colors=n_colors)
        elif cmap == 'blue':
            palette = sns.blend_palette(["#CCE8F9", "#72BBE7", "#4872B4"], n_colors=n_colors)

    color_min, color_max = min(color), max(color)  
    def value_to_color(val):
        color_list = color.tolist()
        color_list.sort()
        if color_min == color_max:
            return palette[-1]
        else:
            index = np.searchsorted(color_list, val, side='left')
            val_position = index / (len(color_list)-1)
            val_position = min(max(val_position, 0), 1)
            ind = int(val_position * (n_colors - 1))
            return palette[ind]
        
    size_min, size_max = min(size), max(size)
    def value_to_size(val):
        size_list = size.tolist()
        size_list.sort()
        if size_min == size_max:
            return 1 * size_scale
        else:
            index = np.searchsorted(size_list, val, side='left')
            val_position = index / (len(size_list)-1)
            val_position = min(max(val_position, 0), 1)
            return val_position * size_scale

    if x_order is not None: 
        x_names = x_order
    else:
        x_names = [t for t in sorted(set([v for v in sender]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if y_order is not None: 
        y_names = y_order
    else:
        y_names = [t for t in sorted(set([v for v in receiver]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    if figsize is None:
        figsize = (len(x_names), len(y_names))

    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    ax.scatter(
        x = [x_to_num[v] for v in sender],
        y = [y_to_num[v] for v in receiver],
        marker = marker,
        s = [value_to_size(v) for v in size], 
        c = [value_to_color(v) for v in color]
    )

    if p_value_plot == True:
        for iter in range(len(sender)):
            isender = sender[iter]
            ireceiver = receiver[iter]
            ipvalue = p_value[iter]
            if ipvalue < p_value_cutoff:
                ax.text(x_to_num[isender], y_to_num[ireceiver], '*', color='black', ha='center', va='center')

    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('white')
    ax.set_xlabel('Sender')
    ax.set_ylabel('Receiver')

    if figsize[0] == figsize[1]:
        ax.set_box_aspect(1)

    if filename is not None:
        plt.savefig(filename)
    
    return ax

def plot_cluster_communication_compare_hierarchy_diagram(
    adata_A: anndata.AnnData,
    adata_B: anndata.AnnData,
    condition_name_A: str,
    condition_name_B: str,
    database_name: str = None,
    clustering: str = None,
    sum_metabolite: str = None,
    sum_metapathway: str = None,
    sum_customerlist: str = None,
    permutation_spatial: bool = False,
    p_value_cutoff: float = 0.05,
    node_sizes_limit: tuple = (50,300),
    edge_sizes_limit: tuple = (0.5,10),
    cmap: str = None,
    alpha: float = 0.5,
    figsize: tuple = (10, 3),
    ax: Optional[mpl.axes.Axes] = None,
    filename: str = None):

    not_none_count = sum(x is not None for x in [sum_metabolite, sum_metapathway, sum_customerlist])
    assert not_none_count < 2, "Please don't not enter all three tasks (sum_metabolite, sum_metapathway, sum_customerlist) at the same time."

    if sum_metabolite is None and sum_metapathway is None and sum_customerlist is None:
        uns_names = 'total-total'
    elif sum_metabolite is not None:
        uns_names = sum_metabolite
    elif sum_metapathway is not None:
        uns_names = sum_metapathway
    elif sum_customerlist is not None:
        uns_names = sum_customerlist

    if permutation_spatial == True:
        culster_name = "MetaChat_cluster_spatial-"  + clustering + "-" + database_name + '-' + uns_names
    else:
        culster_name = "MetaChat_cluster-" + clustering + "-" + database_name + '-' + uns_names

    matrix_condition_A = adata_A.uns[culster_name]['communication_matrix'].copy()
    pvalue_condition_A = adata_A.uns[culster_name]['communication_pvalue'].copy()
    matrix_condition_B = adata_B.uns[culster_name]['communication_matrix'].copy()
    pvalue_condition_B = adata_B.uns[culster_name]['communication_pvalue'].copy()

    if cmap is None:
        cmap = dict(zip(adata_A.obs[clustering].cat.categories.tolist(), adata_A.uns[clustering + '_colors']))

    G_signif = nx.DiGraph()
    G_non_signif = nx.DiGraph()
    if not set(matrix_condition_A.index) == set(matrix_condition_B.index):
        classes = set(matrix_condition_A.index) & set(matrix_condition_B.index)
        print("The group lebel is not the same for the two sets of data, and the intersection will be taken to continue the analysis.")
    else:
        classes = matrix_condition_A.index.tolist()

    node_sizes = {}
    node_colors = {}
    for cls in classes:
        size_L = np.sum(matrix_condition_A, 1)[cls]
        size_M = (np.sum(matrix_condition_A, 0)[cls] + np.sum(matrix_condition_B, 0)[cls])/2
        size_R = np.sum(matrix_condition_B, 1)[cls]
        color = cmap[cls]
        G_signif.add_node(cls + "_L", side='left', size=size_L, color=color)
        G_signif.add_node(cls + "_M", side='middle', size=size_M, color=color)
        G_signif.add_node(cls + "_R", side='right', size=size_R, color=color)
        G_non_signif.add_node(cls + "_L", side='left', size=size_L, color=color)
        G_non_signif.add_node(cls + "_M", side='middle', size=size_M, color=color)
        G_non_signif.add_node(cls + "_R", side='right', size=size_R, color=color)
        node_sizes[cls + "_L"] = size_L
        node_sizes[cls + "_M"] = size_M
        node_sizes[cls + "_R"] = size_R
        node_colors[cls + "_L"] = color
        node_colors[cls + "_M"] = color
        node_colors[cls + "_R"] = color

    node_sizes_min_value = min(node_sizes.values())
    node_sizes_max_value = max(node_sizes.values())

    node_sizes_min_value_new = node_sizes_limit[0]
    node_sizes_max_value_new = node_sizes_limit[1]
    node_sizes_visual = {}

    # 对每个节点的大小进行映射
    for node, size in node_sizes.items():
        # 应用映射公式
        new_size = node_sizes_min_value_new + ((size - node_sizes_min_value) * (node_sizes_max_value_new - node_sizes_min_value_new) / (node_sizes_max_value - node_sizes_min_value))
        node_sizes_visual[node] = new_size

    edges_signif = []
    edges_non_signif = []
    edge_sizes_min_value = np.min([np.min(matrix_condition_A), np.min(matrix_condition_B)])
    edge_sizes_max_value = np.max([np.max(matrix_condition_A), np.max(matrix_condition_B)])
    edge_sizes_min_value_new = edge_sizes_limit[0]
    edge_sizes_max_value_new = edge_sizes_limit[1]

    for cls_sender in classes:
        for cls_receiver in classes:
            weight_A = matrix_condition_A.loc[cls_sender,cls_receiver]
            weight_A = edge_sizes_min_value_new + ((weight_A - edge_sizes_min_value) * (edge_sizes_max_value_new - edge_sizes_min_value_new) / (edge_sizes_max_value - edge_sizes_min_value))
            if pvalue_condition_A.loc[cls_sender,cls_receiver] < p_value_cutoff:
                edges_signif.append((cls_sender + "_L", cls_receiver + "_M", weight_A))
            else:
                edges_non_signif.append((cls_sender + "_L", cls_receiver + "_M", weight_A))

            weight_B = matrix_condition_B.loc[cls_sender,cls_receiver]
            weight_B = edge_sizes_min_value_new + ((weight_B - edge_sizes_min_value) * (edge_sizes_max_value_new - edge_sizes_min_value_new) / (edge_sizes_max_value - edge_sizes_min_value))
            if pvalue_condition_B.loc[cls_sender,cls_receiver] < p_value_cutoff:
                edges_signif.append((cls_sender + "_R", cls_receiver + "_M", weight_B))
            else:
                edges_non_signif.append((cls_sender + "_R", cls_receiver + "_M", weight_B))

    G_signif.add_weighted_edges_from(edges_signif)
    G_non_signif.add_weighted_edges_from(edges_non_signif)

    pos = {}
    for node in G_signif.nodes():
        if '_L' in node:
            pos[node] = (2, len(classes) - classes.index(node[:-2]))
        elif '_M' in node:
            pos[node] = (4, len(classes) - classes.index(node[:-2]))
        else:
            pos[node] = (6, len(classes) - classes.index(node[:-2]))

    # 绘制图形
    fig, ax = plt.subplots(figsize=figsize)
    edges_signif = G_signif.edges(data=True)
    edge_colors_signif = [node_colors[edge[0]] for edge in edges_signif]  # 边的颜色与起始节点颜色一致
    edge_widths_signif = [edge[2]['weight'] for edge in edges_signif]

    edges_non_signif = G_non_signif.edges(data=True)
    edge_colors_non_signif = [node_colors[edge[0]] for edge in edges_non_signif]  # 边的颜色与起始节点颜色一致
    edge_widths_non_signif = [edge[2]['weight'] for edge in edges_non_signif]

    # 绘制节点
    for node in G_signif.nodes():
        if '_M' in node:
            nx.draw_networkx_nodes(G_signif, pos, nodelist=[node], node_color=[node_colors[node]], node_shape='s', node_size=node_sizes_visual[node], ax=ax)
        else:
            nx.draw_networkx_nodes(G_signif, pos, nodelist=[node], node_color=[node_colors[node]], node_size=node_sizes_visual[node], ax=ax)

    labels = {}
    labels_pos = {}
    for cls in classes:
        labels[cls + '_L'] = cls
        labels_pos[cls + '_L'] = (pos[cls + '_L'][0]-0.2, pos[cls + '_L'][1])
    nx.draw_networkx_labels(G_signif, labels_pos, labels=labels, horizontalalignment='right', ax=ax)
    nx.draw_networkx_edges(G_signif, pos, edgelist=edges_signif, edge_color=edge_colors_signif, width=edge_widths_signif, arrowstyle='-|>', arrowsize=10, alpha=1, ax=ax)
    nx.draw_networkx_edges(G_non_signif, pos, edgelist=edges_non_signif, edge_color=edge_colors_non_signif, width=edge_widths_non_signif, arrowstyle='-|>', arrowsize=10, alpha=alpha, ax=ax)

    ax.axis('off')
    ax.set_frame_on(False)
    ax.set_xlim([-1,6.5])
    ax.text(2,len(classes) + 0.8, "Sender", ha='center', va='center', fontsize=12)
    ax.text(4,len(classes) + 0.8, "Receiver", ha='center', va='center', fontsize=12)
    ax.text(6,len(classes) + 0.8, "Sender", ha='center', va='center', fontsize=12)
    ax.arrow(2.35, len(classes) + 0.8, 1.1, 0, head_width=0.3, head_length=0.15, fc='#4F9B79', ec='#4F9B79', linewidth=2)
    ax.arrow(5.65, len(classes) + 0.8, -1.1, 0, head_width=0.3, head_length=0.15, fc='#253071', ec='#253071', linewidth=2)
    ax.text(2.9,len(classes) + 1.4, condition_name_A, ha='center', va='center', fontsize=14) 
    ax.text(5.1,len(classes) + 1.4, condition_name_B, ha='center', va='center', fontsize=14) 

    if filename is not None:
        plt.savefig(filename)

    return ax
  
def plot_MSpair_contribute_cluster(
    adata: anndata.AnnData,
    database_name: str = None,
    clustering: str = None,
    name_metabolite: str = None,
    summary = 'sender',
    cmap: str = None,
    palette = None,
    figsize: tuple = (4,6),
    filename: str = None):

    df_metsen = adata.uns['Metabolite_Sensor_filtered']
    name_sensor = df_metsen.loc[df_metsen['Metabolite'] == name_metabolite,'Sensor'].tolist()

    if summary == 'sender':
        arrv = 's'
    elif summary == 'receiver':
        arrv = 'r'
    ms_pair = [arrv + '-' + name_metabolite + '-' + sensor for sensor in name_sensor]
    ms_pair.sort()

    df_MCC = adata.obsm['MetaChat-' + database_name + '-' + 'sum-' + summary].loc[:, ms_pair].copy()
    df_MCC[clustering] = adata.obs[clustering].copy()

    df_contribute = df_MCC.groupby(clustering).sum()

    n_colors = 256
    if cmap == 'green':
        cmap = sns.blend_palette(["#F4FAFC", "#CAE7E0", "#80C0A5", "#48884B", "#1E4621"], n_colors=n_colors)
    elif cmap == 'blue':
        cmap = sns.blend_palette(["#FAFDFE", "#B7CDE9", "#749FD2", "#4967AC", "#3356A2"], n_colors=n_colors)
    elif cmap == 'red':
        cmap = sns.blend_palette(["#FFFEF7", "#FCF5B8", "#EBA55A", "#C23532"], n_colors=n_colors)

    if palette is None:
        palette_dict = dict(zip(adata.obs[clustering].cat.categories.tolist(), adata.uns[clustering + '_colors']))
        palette = [palette_dict[s] for s in df_contribute.index]

    sns.clustermap(df_contribute.T,
                   row_cluster = False, 
                   col_cluster = False, 
                   col_colors = palette, 
                   cmap = cmap,
                   figsize = figsize,
                   cbar_pos = None)
    
    if filename is not None:
        plt.savefig(filename)

def plot_deg_communication_kegg(
    df_result = None,
    show_term_order = None,
    cmap: str = 'blue',
    maxshow_gene: int = 10,
    figsize = None,  
    filename = None
    ):

    df_show = df_result.iloc[show_term_order,]
    x_names = df_show['Term'].tolist()
    x_names.reverse()

    # x_names = [t for t in sorted(set([v for v in term]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    path_to_genes = {}
    path_to_value = {}
    for index, row in df_show.iterrows():
        gene_list = row['Genes'].split(';')
        genename_show = gene_list[:maxshow_gene]
        genename_show = ';'.join(genename_show)
        path_to_genes[row['Term']] = genename_show
        path_to_value[row['Term']] = -np.log10(row['P-value'])
        
    bar_color = {'blue': '#C9E3F6',
                'green': '#ACD3B7',
                'red': '#F0C3AC'}
    text_color = {'blue': '#2D3A8C',
                'green': '#2E5731',
                'red': '#AD392F'}

    if figsize is not None:
        fig, ax = plt.subplots(figsize = figsize)
    else: 
        fig, ax = plt.subplots()

    x = [x_to_num[v] for v in x_names]
    y = [path_to_value[v] for v in x_names]
    plt.barh(x, y, color=bar_color[cmap], height=0.5) 
    ax.set_facecolor('white')
    for v in x_names:
        ax.text(0+0.05, x_to_num[v], v, color='black', ha='left', va='center')
        ax.text(0+0.05, x_to_num[v]-0.4, path_to_genes[v], color=text_color[cmap], ha='left', va='center')
    ax.set_yticklabels([])
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    ax.set_xticks([t+0.5 for t in ax.get_xticks()], minor=True)
    ax.set_ylim([-0.7, max(x) + 1])
    ax.set_xlabel('-log10(p-value)')
    ax.set_ylabel('KEGG pathway')

    if filename is not None:
        plt.savefig(filename)
    
    return ax

def plot_summary_pathway(ms_result: pd.DataFrame = None,
                         senspathway_rank: pd.DataFrame = None,
                         plot_senspathway_index: list = None,
                         figsize: tuple = (10,10),
                         save_path: str = None
                         ):
    
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.colors as mcolors
    
    # color
    palette_1 = sns.color_palette("tab20",20)
    hex_colors_1 = [mcolors.to_hex(color) for color in palette_1]
    hex_colors_source = [color for index, color in enumerate(hex_colors_1) if index % 2 == 0]
    hex_colors_line = [color for index, color in enumerate(hex_colors_1) if index % 2 == 1]

    palette_2 = sns.color_palette("YlGnBu",len(plot_senspathway_index))
    hex_colors_target = [mcolors.to_hex(color) for color in palette_2][::-1]

    usename_senspathway = list(senspathway_rank.loc[plot_senspathway_index,"Senspathway"])
    ms_result_new = ms_result.loc[:,usename_senspathway]
    
    all_values = ms_result_new.values.flatten()
    non_zero_values = all_values[all_values != 0]
    min_non_zero_value = np.min(non_zero_values)
    ms_result_new = np.log(ms_result_new/min_non_zero_value + 1)
    ms_result_new = ms_result_new.reset_index().copy()
    
    result_all_melted = ms_result_new.melt(id_vars='Metapathway', var_name='Sensorpathway', value_name='communication_score')
    
    metapathway_color = {
        'Metapathway': np.array(ms_result_new.loc[:,"Metapathway"]),
        'color_source': np.array(hex_colors_source[:ms_result_new.shape[0]]),
        'color_link': np.array(hex_colors_line[:ms_result_new.shape[0]])
        }
    metapathway_color = pd.DataFrame(metapathway_color)
    result_all_melted = pd.merge(result_all_melted, metapathway_color, on='Metapathway', how='outer')

    NODES = dict(label = np.concatenate((np.array(ms_result_new.loc[:, "Metapathway"]), 
                                        np.array(usename_senspathway)), axis=0).tolist(),
                color = np.concatenate((np.array(hex_colors_source[:ms_result_new.shape[0]]), 
                                        np.array(hex_colors_target[:len(usename_senspathway)])), axis=0).tolist())

    Node_index = {
        'node': np.concatenate((np.array(ms_result_new.loc[:, "Metapathway"]),
                                np.array(usename_senspathway)), axis=0).tolist(),
        'index': range(ms_result_new.shape[0] + len(usename_senspathway))
    }
    Node_index = pd.DataFrame(Node_index)
    result_all_melted = pd.merge(result_all_melted, Node_index, left_on='Metapathway', right_on = 'node', how='inner')
    result_all_melted = pd.merge(result_all_melted, Node_index, left_on='Sensorpathway', right_on = 'node', how='inner')

    LINKS = dict(source = np.array(result_all_melted["index_x"]).tolist(),
                 target = np.array(result_all_melted["index_y"]).tolist(),
                 value = np.array(result_all_melted["communication_score"]).tolist(),
                 color = np.array(result_all_melted["color_link"]).tolist())

    data = go.Sankey(node = NODES, link = LINKS)
    fig = go.Figure(data)
    fig.show(config={"width": figsize[0], "height": figsize[1]})

    if save_path is not None:
        fig.write_image(save_path, width=figsize[0]*100, height=figsize[1]*100)      