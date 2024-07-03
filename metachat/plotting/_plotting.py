import anndata
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib as mpl
from typing import Optional
from pycirclize import Circos
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as mcolors
 
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
    background_legend: bool = False,
    library_id: str = None,
    group_name: str = None,
    summary: str = "sender",
    cmap: str = "coolwarm",
    group_cmap: dict = None,
    pos_idx: np.ndarray = np.array([0,1],int),
    ndsize: float = 1,
    scale: float = 1.0,
    normalize_summary_quantile: float = 0.995,
    normalize_v: bool = False,
    normalize_v_quantile: float = 0.95,
    arrow_color: str = "#000000",
    grid_density: float = 1.0,
    grid_knn: int = None,
    grid_scale: float = 1.0,
    grid_thresh: float = 1.0,
    grid_width: float = 0.005,
    stream_density: float = 1.0,
    stream_linewidth: float = 1,
    stream_cutoff_perc: float = 5,
    title: str = None,
    plot_savepath: str = None,
    ax: Optional[mpl.axes.Axes] = None
):
    """
    Function for plotting spatial metabolic cell flow.
    
    The metabolic cell communication should have been computed by the function :func:`mc.tl.metabolic_communication`.
    The metabolic cell communication for some specific metabolite, metabolic pathway and customerlist should have been summarized by the function :func:`mc.tl.summary_communication`.
    The metabolic cell communication flow should have been computed by the function :func:`mc.tl.communication_flow`.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    metabolite_name
        Name of a specific metabolite to detect response genes. For example, metabolite_name = 'HMDB0000148'.
    metapathway_name
        Name of a specific metabolic pathways to detect response genes. For example, metabolite_name = 'Alanine, aspartate and glutamate metabolism'.
    customerlist_name
        Name of a specific customerlist to detect response genes. For example, customerlist_name = 'CustomerA'.
    plot_method
        'cell' plot vectors on individual cells. 
        'grid' plot interpolated vectors on regular grids.
        'stream' streamline plot.
    background
        'summary': scatter plot with color representing total sent or received signal.
        'image': the image in Visium data.
        'group': scatter plot with color representing cell groups.
    background_legend
        Whether to include the background legend when background is set to `summary` or `group`.
    library_id
        library_id to specify image when background is set to 'image'.
    group_name
        The key for cell group annotation. Needed if background is set to `group`.
        For example, if ``group_name=='leiden'``, the group annotation should be available in ``.obs['leiden']``.
    summary
        If background is set to 'summary', the numerical value to plot for background.
        'sender': node color represents sender weight.
        'receiver': node color represents receiver weight.
    cmap
        matplotlib colormap name for node summary if numerical (background set to 'summary'), e.g., 'coolwarm'.
        plotly colormap name for node color if summary is (background set to 'group'). e.g., 'Alphabet'.
    group_cmap
        A dictionary that maps group names to colors when setting background to 'group'. If given, ``cmap`` will be ignored.
    pos_idx
        The coordinates to use for plotting (2D plot).
    ndsize
        The node size of the spots.
    scale
        The scale parameter passed to the matplotlib quiver function :func:`matplotlib.pyplot.quiver` for vector field plots.
        The smaller the value, the longer the arrows.
    normalize_summary_quantile
        If background is set to 'summary', the numerical value greater than a quantile will be the same as that quantile.
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
    title
        The title of diagram.
    plot_savepath
        If given, save to the plot_savepath. For example 'mcc_flow.pdf'.
    ax
        An existing matplotlib ax (`matplotlib.axis.Axis`).

    Returns
    -------
    ax : matplotlib.axis.Axis
        The matplotlib ax object of the plot.

    """
    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
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

    if background=='group' and not cmap in ['Plotly','Light24','Dark24','Alphabet']:
        cmap='Alphabet'
    if ax is None:
        fig, ax = plt.subplots()
    if normalize_v:
        V = V / np.quantile(np.linalg.norm(V, axis=1), normalize_v_quantile)
    if group_cmap is None:
        group_cmap = dict(zip(adata.obs[group_name].cat.categories.tolist(), adata.uns[group_name + '_colors']))

    plot_cell_signaling(
        adata.obsm["spatial"][:,pos_idx],
        V,
        signal_sum,
        cmap = cmap,
        group_cmap = group_cmap,
        arrow_color = arrow_color,
        plot_method = plot_method,
        background = background,
        group_name = group_name,
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
        plot_savepath = plot_savepath,
        ax = ax
    )
    return ax


def plot_communication_responseGenes(
    df_deg: pd.DataFrame,
    df_yhat: pd.DataFrame,
    show_gene_names: bool = True,
    top_ngene_per_cluster: int = -1,
    colormap: str = 'magma',
    cluster_colormap: str = 'Plotly',
    color_range: tuple = None,
    font_scale: float = 1,
    figsize: tuple = (10,10),
    plot_savepath = None,
    return_genes = False):
    """
    Plot smoothed gene expression of the detected metabolic cell communication response genes.
    Takes input from the function :func:`metachat.tl.communication_responseGenes` and the function :func:`metachat.tl.communication_responseGenes_cluster`.

    Parameters
    ----------
    df_deg
        A data frame where each row is a gene and the columns should include 'waldStat', 'pvalue', 'cluster'.
        Output of ``mc.tl.communication_responseGenes_cluster``
    df_yhat
        A data frame where each row is the smoothed expression of a gene.
        Output of ``mc.tl.communication_responseGenes_cluster``.
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
    color_range
        If specify, df_yhat will be truncated to this range. For example, color_range = (-2,2).
    font_scale
        Font size.
    figsize
        Fig size to plot the diagram.
    plot_savepath
        Filename for saving the figure. Set the name to end with '.pdf' or 'png' to specify format.   
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
        row_cluster = False, 
        col_cluster = False, 
        col_colors = col_colors,
        cmap = colormap,
        xticklabels = show_gene_names,
        yticklabels = False,
        linewidths = 0,
        figsize = figsize)
    g.ax_heatmap.invert_yaxis()
    g.cax.set_position([.1, .2, .03, .45])
    if plot_savepath is not None:
        plt.savefig(plot_savepath, dpi=300)

    if return_genes:
        return list( df_deg.iloc[idx].index )
    
def plot_group_communication_chord(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    permutation_spatial: bool = False,
    p_value_cutoff: float = 0.05,
    self_communication_off: bool = False,
    highlight_group_sender: str = None,
    highlight_group_receiver: str = None,
    space: int = 5,
    group_cmap: str = None,
    figsize: tuple = (5,5),
    plot_savepath: str = None,
    ax: Optional[mpl.axes.Axes] = None):

    """
    Plot chord diagram for group-level MCC.
    The metabolic cell communication should have been computed by the function :func:`mc.tl.metabolic_communication`.
    The metabolic cell communication for some specific metabolite, metabolic pathway and customerlist should have been summarized by the function :func:`mc.tl.summary_communication`.
    The group-level metabolic cell communication flow should have been computed by the function :func:`mc.tl.communication_group` or :func:`mc.tl.communication_group_spatial`.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    group_name
        Group name of the cell annotation previously saved in ``adata.obs``. 
    metabolite_name
        Name of a specific metabolite to detect response genes. For example, metabolite_name = 'HMDB0000148'.
    metapathway_name
        Name of a specific metabolic pathways to detect response genes. For example, metabolite_name = 'Alanine, aspartate and glutamate metabolism'.
    customerlist_name
        Name of a specific customerlist to detect response genes. For example, customerlist_name = 'CustomerA'.
    permutation_spatial
        Whether to use results from ``mc.tl.communication_group_spatial``.
    p_value_cutoff
        Significance thresholds for Group-level MCC to plot.
    self_communication_off
        Whether to plot self-communication of cell group. If True, self-communication will not display.
    highlight_group_sender
        The group name of sender cells that be highlighted and others group is transparent. Can be specified at the same time as highlight_group_receiver.
    highlight_group_receiver
        The group name of receiver cells that be highlighted and others group is transparent. Can be specified at the same time as highlight_group_sender.
    space
        Distance space between groups.
    group_cmap
        A dictionary that maps group names to colors. 
    plot_savepath
        Filename for saving the figure. Set the name to end with '.pdf' or 'png' to specify format.  
    ax
        An existing matplotlib ax (`matplotlib.axis.Axis`).

    Returns
    -------
    ax : matplotlib.axis.Axis
        The matplotlib ax object of the plot.
    """
    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."
    not_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name])
    assert not_none_count < 2, "Please don't not enter all three tasks (sum_metabolite, sum_metapathway, sum_customerlist) at the same time."

    if metabolite_name is None and metapathway_name is None and customerlist_name is None:
        uns_names = 'total-total'
    elif metabolite_name is not None:
        uns_names = metabolite_name
    elif metapathway_name is not None:
        uns_names = metapathway_name
    elif customerlist_name is not None:
        uns_names = customerlist_name

    if permutation_spatial == True:
        df_communMatrix = adata.uns["MetaChat_group_spatial-"  + group_name + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group_spatial-" + group_name + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()
    else:
        df_communMatrix = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()
    
    df_communMatrix[df_pvalue > p_value_cutoff] = 0
    if self_communication_off:
        for i in range(df_communMatrix.shape[0]):
            df_communMatrix.iloc[i,i] = 0
    df_communMatrix = df_communMatrix.loc[df_communMatrix.sum(axis=1) != 0]
    df_communMatrix = df_communMatrix.loc[:, df_communMatrix.sum(axis=0) != 0]

    link_kws_handler = None
    if (not highlight_group_sender is None) or (not highlight_group_receiver is None):
        def link_kws_handler(from_label: str,
                            to_label: str):
            if (not highlight_group_sender is None) and (highlight_group_receiver is None):
                if from_label in highlight_group_sender:
                    return dict(alpha=0.7, zorder=1.0)
                else:
                    return dict(alpha=0.2, zorder=0)
            elif (highlight_group_sender is None) and (not highlight_group_receiver is None):
                if to_label in highlight_group_receiver:
                    return dict(alpha=0.7, zorder=1.0)
                else:
                    return dict(alpha=0.2, zorder=0)
            else:
                if from_label in highlight_group_sender or to_label in highlight_group_receiver:
                    return dict(alpha=0.7, zorder=1.0)
                else:
                    return dict(alpha=0.2, zorder=0)
    if group_cmap is None:
        group_cmap = dict(zip(adata.obs[group_name].cat.categories.tolist(), adata.uns[group_name + '_colors']))   
   
    if np.sum(np.sum(df_communMatrix)) != 0:
        circos = Circos.initialize_from_matrix(
            df_communMatrix,
            space = space,
            cmap = group_cmap,
            label_kws = dict(size=12),
            link_kws = dict(ec="black", lw=0.5, direction=1),
            link_kws_handler = link_kws_handler
            )
        if plot_savepath is not None:
            circos.savefig(plot_savepath, figsize=figsize)
        else:
            circos.plotfig(figsize=figsize, ax=ax)
    else:
        print("There is no significant group communication in " + uns_names)
    
    return ax

def plot_group_communication_heatmap(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    permutation_spatial: bool = False,
    p_value_plot: bool = True,
    p_value_cutoff: float = 0.05,
    size_scale: int = 300,
    cmap: str = "green",
    palette = None,
    marker: str = 's',
    x_order: list = None, 
    y_order: list = None,
    figsize: tuple = (10,10),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None):

    """
    Plot heatmap diagram for group-level MCC.
    The metabolic cell communication should have been computed by the function :func:`mc.tl.metabolic_communication`.
    The metabolic cell communication for some specific metabolite, metabolic pathway and customerlist should have been summarized by the function :func:`mc.tl.summary_communication`.
    The group-level metabolic cell communication flow should have been computed by the function :func:`mc.tl.communication_group` or :func:`mc.tl.communication_group_spatial`.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    group_name
        Group name of the cell annotation previously saved in ``adata.obs``. 
    metabolite_name
        Name of a specific metabolite to detect response genes. For example, metabolite_name = 'HMDB0000148'.
    metapathway_name
        Name of a specific metabolic pathways to detect response genes. For example, metabolite_name = 'Alanine, aspartate and glutamate metabolism'.
    customerlist_name
        Name of a specific customerlist to detect response genes. For example, customerlist_name = 'CustomerA'.
    permutation_spatial
        Whether to use results from ``mc.tl.communication_group_spatial``.
    p_value_plot:
        Whether to plot significant. if significant, show "*".
    p_value_cutoff
        Significance thresholds for Group-level MCC to plot.
    size_scale
        Control the size of points from values.
    cmap
        color map to plot communication intensity. Can be chose in "green", "red" and "blue".
    palette
        palette to plot communication intensity. Can be generated by "sns.blend_palette". If specify, the "cmap" will not work.
    marker
        The marker shape show in heatmap. It's the same parameter in "ax.scatter".
    x_order
        The order of sender cell groups show in the heatmap. e.g. x_order = ["GroupA", "GroupB", "GroupC"]
    y_order
        The order of receiver cell groups show in the heatmap. e.g. y_order = ["GroupA", "GroupB", "GroupC"]
    figsize
        Fig size to plot the diagram.
    plot_savepath
        Filename for saving the figure. Set the name to end with '.pdf' or 'png' to specify format.   
    ax
        An existing matplotlib ax (`matplotlib.axis.Axis`).

    Returns
    -------
    ax : matplotlib.axis.Axis
        The matplotlib ax object of the plot.

    """ 

   # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."
    not_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name])
    assert not_none_count < 2, "Please don't not enter all three tasks (sum_metabolite, sum_metapathway, sum_customerlist) at the same time."

    if metabolite_name is None and metapathway_name is None and customerlist_name is None:
        uns_names = 'total-total'
    elif metabolite_name is not None:
        uns_names = metabolite_name
    elif metapathway_name is not None:
        uns_names = metapathway_name
    elif customerlist_name is not None:
        uns_names = customerlist_name

    if permutation_spatial == True:
        df_communMatrix = adata.uns["MetaChat_group_spatial-"  + group_name + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group_spatial-" + group_name + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()
    else:
        df_communMatrix = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + uns_names]['communication_pvalue'].copy()

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

    if plot_savepath is not None:
        plt.savefig(plot_savepath)
    
    return ax

def plot_group_communication_compare_hierarchy_diagram(
    adata_A: anndata.AnnData,
    adata_B: anndata.AnnData,
    condition_name_A: str = None,
    condition_name_B: str = None,
    database_name: str = None,
    group_name: str = None,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    permutation_spatial: bool = False,
    p_value_cutoff: float = 0.05,
    node_sizes_limit: tuple = (50,300),
    edge_sizes_limit: tuple = (0.5,10),
    group_cmap: str = None,
    alpha: float = 0.5,
    figsize: tuple = (10,3),
    plot_savepath: str = None,
    ax: Optional[mpl.axes.Axes] = None):

    """
    Plot hierarachy diagram for comparing group-level MCC between two conditions.
    The metabolic cell communication should have been computed by the function :func:`mc.tl.metabolic_communication`.
    The metabolic cell communication for some specific metabolite, metabolic pathway and customerlist should have been summarized by the function :func:`mc.tl.summary_communication`.
    The group-level metabolic cell communication flow should have been computed by the function :func:`mc.tl.communication_group` or :func:`mc.tl.communication_group_spatial`.

    Parameters
    ----------
    adata_A
        The data matrix of shape ``n_obs`` × ``n_var`` for condition A.
        Rows correspond to cells or positions and columns to genes.
    adata_B
        The data matrix of shape ``n_obs`` × ``n_var`` for condition B.
        Rows correspond to cells or positions and columns to genes.
    condition_name_A
        Name of condition A.
    condition_name_B
        Name of condition B.   
    database_name
        Name of the Metabolite-Sensor interaction database.
    group_name
        Group name of the cell annotation previously saved in ``adata.obs``. 
    metabolite_name
        Name of a specific metabolite to detect response genes. For example, metabolite_name = 'HMDB0000148'.
    metapathway_name
        Name of a specific metabolic pathways to detect response genes. For example, metabolite_name = 'Alanine, aspartate and glutamate metabolism'.
    customerlist_name
        Name of a specific customerlist to detect response genes. For example, customerlist_name = 'CustomerA'.
    permutation_spatial
        Whether to use results from ``mc.tl.communication_group_spatial``.
    p_value_cutoff
        Significance thresholds for Group-level MCC to plot.
    node_sizes_limit
        Control the limit of the node size. 
    edge_sizes_limit
        Control the limit of the edge size. 
    group_cmap
        A dictionary that maps group names to colors. 
    alpha 
        Transparency of insignificant edges
    figsize
        Fig size to plot the diagram.
    plot_savepath
        Filename for saving the figure. Set the name to end with '.pdf' or 'png' to specify format.   
    ax
        An existing matplotlib ax (`matplotlib.axis.Axis`).

    Returns
    -------
    ax : matplotlib.axis.Axis
        The matplotlib ax object of the plot.

    """ 
    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."
    not_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name])
    assert not_none_count < 2, "Please don't not enter all three tasks (sum_metabolite, sum_metapathway, sum_customerlist) at the same time."

    if metabolite_name is None and metapathway_name is None and customerlist_name is None:
        uns_names = 'total-total'
    elif metabolite_name is not None:
        uns_names = metabolite_name
    elif metapathway_name is not None:
        uns_names = metapathway_name
    elif customerlist_name is not None:
        uns_names = customerlist_name

    if permutation_spatial == True:
        culster_name = "MetaChat_group_spatial-"  + group_name + "-" + database_name + '-' + uns_names
    else:
        culster_name = "MetaChat_group-" + group_name + "-" + database_name + '-' + uns_names

    matrix_condition_A = adata_A.uns[culster_name]['communication_matrix'].copy()
    pvalue_condition_A = adata_A.uns[culster_name]['communication_pvalue'].copy()
    matrix_condition_B = adata_B.uns[culster_name]['communication_matrix'].copy()
    pvalue_condition_B = adata_B.uns[culster_name]['communication_pvalue'].copy()

    if group_cmap is None:
        group_cmap = dict(zip(adata_A.obs[group_name].cat.categories.tolist(), adata_A.uns[group_name + '_colors']))

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
        color = group_cmap[cls]
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

    for node, size in node_sizes.items():
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

    # plot diagram
    fig, ax = plt.subplots(figsize=figsize)
    edges_signif = G_signif.edges(data=True)
    edge_colors_signif = [node_colors[edge[0]] for edge in edges_signif]
    edge_widths_signif = [edge[2]['weight'] for edge in edges_signif]

    edges_non_signif = G_non_signif.edges(data=True)
    edge_colors_non_signif = [node_colors[edge[0]] for edge in edges_non_signif]
    edge_widths_non_signif = [edge[2]['weight'] for edge in edges_non_signif]

    # plot nodes
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

    if plot_savepath is not None:
        plt.savefig(plot_savepath)

    return ax
 
def plot_MSpair_contribute_group(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    metabolite_name: str = None,
    summary: str = 'sender',
    cmap: str = "green",
    group_cmap = None,
    figsize: tuple = (4,6),
    plot_savepath: str = None):

    """
    Plot heatmap diagram for comparing contribution.
    The metabolic cell communication should have been computed by the function :func:`mc.tl.metabolic_communication`.
    The metabolic cell communication for some specific metabolite, metabolic pathway and customerlist should have been summarized by the function :func:`mc.tl.summary_communication`.
    The group-level metabolic cell communication should have been computed by the function :func:`mc.tl.communication_group` or :func:`mc.tl.communication_group_spatial`.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the Metabolite-Sensor interaction database.
    group_name
        Group name of the cell annotation previously saved in ``adata.obs``. 
    metabolite_name
        Name of a specific metabolite to detect response genes. For example, metabolite_name = 'HMDB0000148'.
    summary
        The sender signals or receiver signals that be computed contribution.
    cmap
        color map to plot communication intensity. Can be chose in "green", "red" and "blue".
    group_cmap
        A dictionary that maps group names to colors.    
    figsize
        Fig size to plot the diagram.
    plot_savepath
        Filename for saving the figure. Set the name to end with '.pdf' or 'png' to specify format.   

    """ 

    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."
    assert metabolite_name is not None, "Please at least specify metabolite_name."

    df_metasen = adata.uns['Metabolite_Sensor_filtered']
    name_sensor = df_metasen.loc[df_metasen['Metabolite'] == metabolite_name,'Sensor'].tolist()

    if summary == 'sender':
        arrv = 's'
    elif summary == 'receiver':
        arrv = 'r'
    ms_pair = [arrv + '-' + metabolite_name + '-' + sensor for sensor in name_sensor]
    ms_pair.sort()

    df_MCC = adata.obsm['MetaChat-' + database_name + '-' + 'sum-' + summary].loc[:, ms_pair].copy()
    df_MCC[group_name] = adata.obs[group_name].copy()
    df_contribute = df_MCC.groupby(group_name).sum()

    n_colors = 256
    if cmap == 'green':
        cmap = sns.blend_palette(["#F4FAFC", "#CAE7E0", "#80C0A5", "#48884B", "#1E4621"], n_colors=n_colors)
    elif cmap == 'blue':
        cmap = sns.blend_palette(["#FAFDFE", "#B7CDE9", "#749FD2", "#4967AC", "#3356A2"], n_colors=n_colors)
    elif cmap == 'red':
        cmap = sns.blend_palette(["#FFFEF7", "#FCF5B8", "#EBA55A", "#C23532"], n_colors=n_colors)

    if group_cmap is None:
        group_cmap_dict = dict(zip(adata.obs[group_name].cat.categories.tolist(), adata.uns[group_name + '_colors']))
        group_cmap = [group_cmap_dict[s] for s in df_contribute.index]

    sns.clustermap(df_contribute.T,
                   row_cluster = False, 
                   col_cluster = False, 
                   col_colors = group_cmap, 
                   cmap = cmap,
                   figsize = figsize,
                   cbar_pos = None)
    
    if plot_savepath is not None:
        plt.savefig(plot_savepath)

def plot_communication_responseGenes_keggEnrich(
    df_result: pd.DataFrame,
    show_term_order: list = [0,1,2,3,4],
    cmap: str = 'green',
    maxshow_gene: int = 10,
    figsize: tuple = (6,6),  
    plot_savepath: str = None):

    """
    Barplot for results of KEGG enrichment by MCC response genes
    
    Parameters
    ----------
    df_result
        The results from the function :func:`mc.tl.communication_responseGenes`
    show_term_order : list, optional
        List of terms to be shown in the plot, in the order specified. If None, the top 5 terms will be shown.
    cmap
        Color map to use for the barplot. It can be chose in 'green', 'blue', 'red'.
    maxshow_gene
        Maximum number of genes to show in the plot. Default is 10.
    figsize
        Size of the figure. Default is (6, 6).
    plot_savepath
        Path to save the plot. If None, the plot will be shown but not saved.
    """

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

    if plot_savepath is not None:
        plt.savefig(plot_savepath)

def plot_summary_pathway(ms_result: pd.DataFrame = None,
                         senspathway_rank: pd.DataFrame = None,
                         plot_senspathway_index: list = None,
                         figsize: tuple = (10,10),
                         plot_savepath: str = None):
    """
    Function to plot a Sankey diagram summarizing the metabolic cell communication pathways.

    Parameters
    ----------
    ms_result
        A DataFrame containing the metabolic communication scores between different pathways.
    senspathway_rank
        A DataFrame containing the ranking of sensor pathways.
    plot_senspathway_index
        A list of indices specifying which sensor pathways to include in the plot.
    figsize
        Size of the figure in inches. Default is (10, 10).
    plot_savepath
        Path to save the plot image. If None, the plot will be displayed but not saved.
    """

    # color
    palette_1 = sns.color_palette("tab20",20)
    hex_colors_1 = [mcolors.to_hex(color) for color in palette_1]
    hex_colors_source = [color for index, color in enumerate(hex_colors_1) if index % 2 == 0]
    hex_colors_line = [color for index, color in enumerate(hex_colors_1) if index % 2 == 1]

    palette_2 = sns.color_palette("YlGnBu",len(plot_senspathway_index))
    hex_colors_target = [mcolors.to_hex(color) for color in palette_2][::-1]

    usename_senspathway = list(senspathway_rank.loc[plot_senspathway_index,"Sensor.Pathway"])
    ms_result_new = ms_result.loc[:,usename_senspathway]
    
    all_values = ms_result_new.values.flatten()
    non_zero_values = all_values[all_values != 0]
    min_non_zero_value = np.min(non_zero_values)
    ms_result_new = np.log(ms_result_new/min_non_zero_value + 1)
    ms_result_new = ms_result_new.reset_index().copy()
    
    result_all_melted = ms_result_new.melt(id_vars='Metabolite.Pathway', var_name='Sensor.Pathway', value_name='communication_score')
    
    metapathway_color = {
        'Metabolite.Pathway': np.array(ms_result_new.loc[:,"Metabolite.Pathway"]),
        'color_source': np.array(hex_colors_source[:ms_result_new.shape[0]]),
        'color_link': np.array(hex_colors_line[:ms_result_new.shape[0]])
        }
    metapathway_color = pd.DataFrame(metapathway_color)
    result_all_melted = pd.merge(result_all_melted, metapathway_color, on='Metabolite.Pathway', how='outer')

    NODES = dict(label = np.concatenate((np.array(ms_result_new.loc[:, "Metabolite.Pathway"]), 
                                        np.array(usename_senspathway)), axis=0).tolist(),
                color = np.concatenate((np.array(hex_colors_source[:ms_result_new.shape[0]]), 
                                        np.array(hex_colors_target[:len(usename_senspathway)])), axis=0).tolist())

    Node_index = {
        'node': np.concatenate((np.array(ms_result_new.loc[:, "Metabolite.Pathway"]),
                                np.array(usename_senspathway)), axis=0).tolist(),
        'index': range(ms_result_new.shape[0] + len(usename_senspathway))
    }
    Node_index = pd.DataFrame(Node_index)
    result_all_melted = pd.merge(result_all_melted, Node_index, left_on='Metabolite.Pathway', right_on = 'node', how='inner')
    result_all_melted = pd.merge(result_all_melted, Node_index, left_on='Sensor.Pathway', right_on = 'node', how='inner')

    LINKS = dict(source = np.array(result_all_melted["index_x"]).tolist(),
                 target = np.array(result_all_melted["index_y"]).tolist(),
                 value = np.array(result_all_melted["communication_score"]).tolist(),
                 color = np.array(result_all_melted["color_link"]).tolist())

    data = go.Sankey(node = NODES, link = LINKS)
    fig = go.Figure(data)
    fig.show(config={"width": figsize[0], "height": figsize[1]})

    if plot_savepath is not None:
        fig.write_image(plot_savepath, width=figsize[0]*100, height=figsize[1]*100)      