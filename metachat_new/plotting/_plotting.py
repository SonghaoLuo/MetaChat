from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import distance_matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as patches
import matplotlib.lines as mlines

import seaborn as sns
import squidpy as sq
import plotly.express as px
import plotly.graph_objects as go
from pycirclize import Circos
from adjustText import adjust_text
from plotly.subplots import make_subplots

import networkx as nx
import anndata

from .._utils import plot_cell_signaling
from .._utils import get_cmap_qualitative

def plot_communication_flow(
    adata: anndata.AnnData,
    database_name: str,
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    ms_pair_name: str = None,
    summary: str = "receiver",
    plot_method: str = "grid",
    background: str = "image",
    library_id: str = None,
    group_name: str = None,
    group_cmap: dict = None,
    background_legend: bool = False,
    cmap: str = "coolwarm",
    pos_idx: np.ndarray = np.array([0,1], int),
    normalize_summary_quantile: float = 0.995,
    ndsize: float = 3,
    vmin=None,
    vmax=None,
    grid_density: float = 1.0,
    grid_knn: int = None,
    grid_scale: float = 1.0,
    grid_thresh: float = 1.0,
    arrow_color: str = "#000000",
    arrow_width: float = 0.005,
    largest_arrow: float = 0.05,
    normalize_v: bool = False,
    normalize_v_quantile: float = 0.95,
    title: str = None,
    plot_savepath: str = None,
    ax: Optional[mpl.axes.Axes] = None
):
    """
    Visualize inferred metabolic communication vector fields on tissue images or annotated backgrounds.

    This function overlays communication flow vectors computed from MetaChat analysis results on tissue images,
    categorical annotations, or summary backgrounds. It supports spot-wise, grid-interpolated, or streamline rendering.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing spatial coordinates in ``adata.obsm["spatial"]``.
    database_name : str
        Name of the metabolite–sensor interaction database used for communication inference.
    metabolite_name : str, optional
        Name of a specific metabolite to visualize. Mutually exclusive with ``metapathway_name``,
        ``customerlist_name``, and ``ms_pair_name``.
    metapathway_name : str, optional
        Name of a specific metabolic pathway to visualize.
    customerlist_name : str, optional
        Name of a custom metabolite–sensor list to visualize.
    ms_pair_name : str, optional
        Name of a metabolite–sensor pair to visualize.
    summary : {"sender", "receiver"}, default="receiver"
        Type of summary statistic used for background coloring.
    plot_method : {"cell", "grid", "stream"}, default="grid"
        Rendering mode for vector visualization.
    background : {"summary", "image", "group"}, default="image"
        Type of background to draw behind the vectors.
    library_id : str, optional
        Visium library identifier used when ``background="image"``.
    group_name : str, optional
        Column in ``adata.obs`` used for categorical coloring when ``background="group"``.
    group_cmap : dict, optional
        Mapping from category names to colors.
    background_legend : bool, default=False
        Whether to display legend for the background.
    cmap : str, default="coolwarm"
        Colormap name for summary or numeric backgrounds.
    pos_idx : np.ndarray, default=np.array([0,1])
        Indices of spatial coordinates used for plotting.
    normalize_summary_quantile : float, default=0.995
        Quantile for clipping background values.
    ndsize : float, default=3
        Spot marker size.
    vmin, vmax : float, optional
        Color scale limits for background values.
    grid_density : float, default=1.0
        Density of interpolation grid when ``plot_method="grid"``.
    grid_knn : int, optional
        Number of neighbors used for vector interpolation.
    grid_scale : float, default=1.0
        Kernel scale factor for interpolation.
    grid_thresh : float, default=1.0
        Minimum interpolation weight threshold.
    arrow_color : str, default="#000000"
        Color of flow arrows.
    arrow_width : float, default=0.005
        Width of quiver arrows.
    largest_arrow : float, default=0.05
        Maximum arrow length after normalization.
    normalize_v : bool, default=False
        Whether to normalize vector magnitudes for visualization.
    normalize_v_quantile : float, default=0.95
        Quantile for magnitude clipping before normalization.
    title : str, optional
        Title of the plot.
    plot_savepath : str, optional
        File path to save the plot; no saving if ``None``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw the plot on; creates new one if ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the resulting plot.
    coords_plot : np.ndarray
        Coordinates used for plotting (after interpolation).
    V_plot : np.ndarray
        Vector field used for plotting (after interpolation).
    """

    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    not_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name, ms_pair_name])
    assert not_none_count <= 1, "Please specify at most one of metabolite_name, metapathway_name, customerlist_name, or ms_pair_name."

    if summary == 'sender':
        summary_abbr = 's'
    elif summary == 'receiver':
        summary_abbr = 'r'
    else:
        raise ValueError("summary must be either 'sender' or 'receiver'.")

    if metabolite_name is None and metapathway_name is None and customerlist_name is None and ms_pair_name is None:
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
    elif ms_pair_name is not None:
        vf_name = ms_pair_name
        sum_name = ms_pair_name
        obsm_name = ''

    V = adata.obsm['MetaChat-vf-' + database_name + '-' + summary + '-' + vf_name][:,pos_idx].copy()
    signal_sum = adata.obsm['MetaChat-' + database_name + "-sum-" + summary + obsm_name][summary_abbr + '-' + sum_name].copy()

    if background=='group':
        if group_cmap is None:
            group_cmap = dict(zip(adata.obs[group_name].cat.categories.tolist(), adata.uns[group_name + '_colors']))
        elif not group_cmap.lower() in ['plotly','light24','dark24','alphabet']:
            group_cmap='alphabet'

    if ax is None:
        fig, ax = plt.subplots()
    if normalize_v:
        V = V / np.quantile(np.linalg.norm(V, axis=1), normalize_v_quantile)
  
    ax, coords_plot, V_plot = plot_cell_signaling(
        coords = adata.obsm["spatial"][:,pos_idx],
        Vector = V,
        signal_sum = signal_sum,
        summary = summary,
        plot_method = plot_method,
        adata = adata,
        background = background,
        library_id = library_id,
        group_name = group_name,
        group_cmap = group_cmap,
        background_legend = background_legend,
        cmap = cmap,
        normalize_summary_quantile = normalize_summary_quantile,
        ndsize = ndsize,
        vmin = vmin,
        vmax = vmax,
        grid_density = grid_density,
        grid_knn = grid_knn,
        grid_scale = grid_scale,
        grid_thresh = grid_thresh,
        arrow_color = arrow_color,
        arrow_width = arrow_width,
        largest_arrow = largest_arrow,      
        title = title,
        plot_savepath = plot_savepath,
        ax = ax
    )

    return ax, coords_plot, V_plot
    
def plot_group_communication_chord(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    summary: str = 'sender',
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    ms_pair_name: str = None,
    permutation_spatial: bool = False,
    p_value_cutoff: float = 0.05,
    self_communication_off: bool = False,
    highlight_group_sender: str = None,
    highlight_group_receiver: str = None,
    space: int = 5,
    group_cmap: str = None,
    figsize: tuple = (5, 5),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None,
):

    """
    Plot a chord diagram representing group-level metabolic cell communication (MCC).

    This function visualizes inter-group communication strength as a chord diagram, 
    based on results computed by MetaChat group-level analysis. Each arc represents 
    a cell group, and links represent the strength of metabolite-mediated communication 
    between sender and receiver groups.

    The communication matrix and associated p-values must be precomputed by
    :func:`mc.tl.communication_group` or :func:`mc.tl.communication_group_spatial`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing group-level MCC results in ``adata.uns``.
    database_name : str
        Name of the metabolite–sensor interaction database used for MCC analysis.
    group_name : str
        Column name in ``adata.obs`` specifying cell group annotations.
    summary : {"sender", "receiver"}, default="sender"
        Direction of communication summary to visualize.
    metabolite_name : str, optional
        Name of a specific metabolite to visualize.
    metapathway_name : str, optional
        Name of a specific metabolic pathway to visualize.
    customerlist_name : str, optional
        Name of a specific custom metabolite list to visualize.
    ms_pair_name: str, optional
        Name of a specific metabolite–sensor pair to visualize.
    permutation_spatial : bool, default=False
        Whether to use results from spatially permuted communication tests
        (``mc.tl.communication_group_spatial``).
    p_value_cutoff : float, default=0.05
        P-value threshold for filtering significant group-level communications.
    self_communication_off : bool, default=False
        Whether to remove self-communication.
    highlight_group_sender : str or list of str, optional
        Name(s) of sender group(s) to highlight. Other groups are rendered transparent.
    highlight_group_receiver : str or list of str, optional
        Name(s) of receiver group(s) to highlight. Other groups are rendered transparent.
    space : int, default=5
        Angular spacing between group arcs in degrees.
    group_cmap : dict or str, optional
        Mapping from group names to colors, or qualitative palette name
        (e.g., ``'Plotly'``, ``'Alphabet'``).
    figsize : tuple of float, default=(5, 5)
        Figure size (width, height).
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw the chord diagram. If ``None``, a new one is created.
    plot_savepath : str, optional
        File path to save the resulting figure. The format is inferred from the file extension.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the chord diagram.

    Notes
    -----
    - The function requires that group-level communication results are already computed and stored in:
        - ``adata.uns["MetaChat_group-<group_name>-<database_name>-<summary>-<target>"]``  
          (for :func:`mc.tl.communication_group`), or
        - ``adata.uns["MetaChat_group_spatial-..."]``  
          (for :func:`mc.tl.communication_group_spatial`).

    - Links are filtered by ``p_value_cutoff`` and optionally exclude self-communications.

    - Highlight options can be used to emphasize communication from or to selected cell groups.

    - This visualization helps identify dominant sender–receiver relationships across cell types or regions.
    """
    
    # Check inputs
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."
    not_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name, ms_pair_name])
    assert not_none_count <= 1, ("Please specify at most one of metabolite_name, metapathway_name, customerlist_name, or ms_pair_name.")

    if metabolite_name:
        uns_names = metabolite_name
    elif metapathway_name:
        uns_names = metapathway_name
    elif customerlist_name:
        uns_names = customerlist_name
    elif ms_pair_name:
        uns_names = ms_pair_name
    else:
        uns_names = "total-total"

    if permutation_spatial == True:
        df_communMatrix = adata.uns["MetaChat_group_spatial-"  + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group_spatial-" + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_pvalue'].copy()
    else:
        df_communMatrix = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_pvalue'].copy()

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
    ms_pair_name: str = None,
    summary: str = "sender",
    permutation_spatial: bool = False,
    p_value_plot: bool = True,
    p_value_cutoff: float = 0.05,
    size_scale: float = 300,
    cmap: str = "green",
    palette: Optional[list] = None,
    marker: str = "s",
    x_order: Optional[list] = None,
    y_order: Optional[list] = None,
    figsize: tuple = (10, 10),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: Optional[str] = None
):

    """
    Plot a heatmap diagram for group-level metabolic cell communication (MCC).

    This function visualizes group-level MCC intensity between sender and receiver cell groups,
    using aheatmap representation. Each point corresponds to a sender–receiver pair,
    colored by communication strength and optionally marked with significance (*).

    The group-level communication results must be computed beforehand by
    :func:`mc.tl.communication_group` or :func:`mc.tl.communication_group_spatial`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing group-level MCC results in ``adata.uns``.
    database_name : str
        Name of the metabolite–sensor interaction database.
    group_name : str
        Column name in ``adata.obs`` specifying the cell group annotation.
    metabolite_name : str, optional
        Name of a specific metabolite to visualize (e.g., ``'HMDB0000148'``).
    metapathway_name : str, optional
        Name of a specific metabolic pathway to visualize.
    customerlist_name : str, optional
        Name of a custom metabolite list to visualize.
    ms_pair_name : str, optional
        Name of a specific metabolite–sensor pair to visualize.
    summary : {"sender", "receiver"}, default="sender"
        Direction of communication summary to visualize.
    permutation_spatial : bool, default=False
        Whether to use results from ``mc.tl.communication_group_spatial``.
    p_value_plot : bool, default=True
        Whether to mark significant interactions with an asterisk ``*``.
    p_value_cutoff : float, default=0.05
        Significance threshold for group-level MCC visualization.
    size_scale : float, default=300
        Scaling factor controlling marker size proportional to MCC score.
    cmap : {"green", "red", "blue"}, default="green"
        Predefined color theme for communication strength.
    palette : list, optional
        Custom color palette list. If specified, overrides ``cmap``.
    marker : str, default="s"
        Matplotlib marker style used in the scatter plot.
    x_order : list of str, optional
        Order of sender groups along the x-axis.
    y_order : list of str, optional
        Order of receiver groups along the y-axis.
    figsize : tuple of float, default=(10, 10)
        Size of the output figure (width, height).
    plot_savepath : str, optional
        File path to save the figure. The format is inferred from the extension.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. If ``None``, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the heatmap plot.

    Notes
    -----
    - The function expects communication matrices and p-values stored in:
        - ``adata.uns["MetaChat_group-<group_name>-<database_name>-<summary>-<target>"]`` or  
        - ``adata.uns["MetaChat_group_spatial-..."]`` for spatial permutation results.
    - Significance marks (``*``) are added when ``p_value_plot=True`` and p < ``p_value_cutoff``.
    - Marker size and color are scaled by communication intensity.
    - Use ``x_order`` and ``y_order`` to enforce custom group display order.

    """ 

   # ==== Check inputs ====
    assert database_name is not None, "Please at least specify database_name."
    assert group_name is not None, "Please at least specify group_name."

    not_none_count = sum(
        x is not None 
        for x in [metabolite_name, metapathway_name, customerlist_name, ms_pair_name]
    )

    assert not_none_count <= 1, (
        "Please specify at most one of metabolite_name, metapathway_name, "
        "customerlist_name, or ms_pair_name."
    )
    
    if metabolite_name:
        uns_names = metabolite_name
    elif metapathway_name:
        uns_names = metapathway_name
    elif customerlist_name:
        uns_names = customerlist_name
    elif ms_pair_name:
        uns_names = ms_pair_name
    else:
        uns_names = "total-total"
    
    # ==== Load communication results ====
    if permutation_spatial == True:
        df_communMatrix = adata.uns["MetaChat_group_spatial-"  + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group_spatial-" + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_pvalue'].copy()
    else:
        df_communMatrix = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_matrix'].copy()
        df_pvalue = adata.uns["MetaChat_group-" + group_name + "-" + database_name + '-' + summary + '-' + uns_names]['communication_pvalue'].copy()

    # ==== Prepare plot settings ====
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
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ==== Plot heatmap ====
    ax.scatter(
        x = [x_to_num[v] for v in sender],
        y = [y_to_num[v] for v in receiver],
        marker = marker,
        s = [value_to_size(v) for v in size], 
        c = [value_to_color(v) for v in color]
    )

    norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
    cmap_mpl = mpl.colors.ListedColormap(palette)
    sm = ScalarMappable(cmap=cmap_mpl, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MCC Score", fontsize=10)

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
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")
    
    return ax

def plot_group_communication_compare_hierarchy_diagram(
    adata_A: anndata.AnnData,
    adata_B: anndata.AnnData,
    condition_name_A: str = None,
    condition_name_B: str = None,
    database_name: str = None,
    group_name: str = None,
    summary: str = 'sender',
    metabolite_name: str = None,
    metapathway_name: str = None,
    customerlist_name: str = None,
    ms_pair_name: str = None,
    permutation_spatial: bool = False,
    p_value_cutoff: float = 0.05,
    node_sizes_limit: tuple = (50, 300),
    edge_sizes_limit: tuple = (0.5,10),
    group_cmap: dict = None,
    alpha: float = 0.5,
    figsize: tuple = (10, 3),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None,
):
    """
    Plot a hierarchy-style diagram comparing group-level MCC between two conditions.

    This visualization contrasts the direction and strength of group-level metabolic
    cell communication (MCC) between two conditions (e.g., Control vs Disease).
    Sender–receiver relationships are represented as directed edges between nodes
    on two parallel panels, showing both conditions side by side.

    Parameters
    ----------
    adata_A : anndata.AnnData
        Annotated data matrix for condition A. Must contain group-level MCC results.
    adata_B : anndata.AnnData
        Annotated data matrix for condition B. Must contain the same structure as ``adata_A``.
    condition_name_A : str, optional
        Label for condition A, shown on the left panel (e.g., "Control").
    condition_name_B : str, optional
        Label for condition B, shown on the right panel (e.g., "Disease").
    database_name : str
        Name of the metabolite–sensor interaction database used for MCC computation.
    group_name : str
        Column name in ``adata.obs`` containing group/cell-type annotations.
    summary : {"sender", "receiver"}, default="sender"
        Direction of communication summary to visualize.
    metabolite_name : str, optional
        Specific metabolite to visualize.
    metapathway_name : str, optional
        Specific metabolic pathway to visualize.
    customerlist_name : str, optional
        Custom metabolite list name to visualize.
    ms_pair_name : str, optional
        Name of a specific metabolite–sensor pair to visualize.
    permutation_spatial : bool, default=False
        Whether to use spatially permuted results from :func:`mc.tl.communication_group_spatial`.
    p_value_cutoff : float, default=0.05
        Threshold for significance in group-level communication.
    node_sizes_limit : tuple of float, default=(50, 300)
        Minimum and maximum node size for scaling group strength.
    edge_sizes_limit : tuple of float, default=(0.5, 10)
        Minimum and maximum edge width for scaling communication intensity.
    group_cmap : dict, optional
        Mapping from group names to colors. If ``None``, retrieved from
        ``adata.obs[group_name].cat.categories`` and corresponding color table in ``adata.uns[group_name + "_colors"]``.
    alpha : float, default=0.5
        Transparency for non-significant edges.
    figsize : tuple of float, default=(10, 3)
        Figure size (width, height).
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw the hierarchy diagram on.  
        If ``None``, a new figure and axis are created.
    plot_savepath : str, optional
        File path to save the figure (e.g., ``"results/group_compare_hierarchy.pdf"``).  
        The format is inferred from the file extension. If ``None``, the plot is displayed
        interactively without saving.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the hierarchy comparison diagram.

    Notes
    -----
    - This function visualizes the output from :func:`mc.tl.communication_group`
      or :func:`mc.tl.communication_group_spatial` for two different conditions.
    - Node size reflects total communication strength per group.
    - Edge width and color indicate direction and magnitude of communication.
    - Non-significant edges (p ≥ cutoff) are rendered transparent with reduced opacity.

    """

    # ==== Check inputs ====
    assert database_name is not None, "Please specify `database_name`."
    assert group_name is not None, "Please specify `group_name`."
    not_none_count = sum(x is not None for x in [metabolite_name, metapathway_name, customerlist_name, ms_pair_name])
    assert not_none_count <= 1, (
        "Please specify at most one of metabolite_name, metapathway_name, customerlist_name or ms_pair_name."
    )

    if metabolite_name:
        uns_names = metabolite_name
    elif metapathway_name:
        uns_names = metapathway_name
    elif customerlist_name:
        uns_names = customerlist_name
    elif ms_pair_name:
        uns_names = ms_pair_name
    else:
        uns_names = "total-total"

    if permutation_spatial == True:
        culster_name = "MetaChat_group_spatial-"  + group_name + "-" + database_name + '-' + summary + '-' + uns_names
    else:
        culster_name = "MetaChat_group-" + group_name + "-" + database_name + '-' + summary + '-' + uns_names
    
    # ==== Prepare data ====
    matrix_condition_A = adata_A.uns[culster_name]['communication_matrix'].copy()
    pvalue_condition_A = adata_A.uns[culster_name]['communication_pvalue'].copy()
    matrix_condition_B = adata_B.uns[culster_name]['communication_matrix'].copy()
    pvalue_condition_B = adata_B.uns[culster_name]['communication_pvalue'].copy()
    
    classes = sorted(set(matrix_condition_A.index).union(set(matrix_condition_B.index)))

    expanded_matrix_condition_A = pd.DataFrame(0, index=classes, columns=classes)
    expanded_matrix_condition_B = pd.DataFrame(0, index=classes, columns=classes)
    expanded_matrix_condition_A.loc[matrix_condition_A.index, matrix_condition_A.columns] = matrix_condition_A
    expanded_matrix_condition_B.loc[matrix_condition_B.index, matrix_condition_B.columns] = matrix_condition_B

    expanded_pvalue_condition_A = pd.DataFrame(1, index=classes, columns=classes)
    expanded_pvalue_condition_B = pd.DataFrame(1, index=classes, columns=classes)
    expanded_pvalue_condition_A.loc[pvalue_condition_A.index, pvalue_condition_A.columns] = pvalue_condition_A
    expanded_pvalue_condition_B.loc[pvalue_condition_B.index, pvalue_condition_B.columns] = pvalue_condition_B

    matrix_condition_A = expanded_matrix_condition_A.copy()
    pvalue_condition_A = expanded_pvalue_condition_A.copy()
    matrix_condition_B = expanded_matrix_condition_B.copy()
    pvalue_condition_B = expanded_pvalue_condition_B.copy()

    if group_cmap is None:
        group_cmap_A = dict(zip(adata_A.obs[group_name].cat.categories.tolist(), adata_A.uns[group_name + '_colors']))
        group_cmap_B = dict(zip(adata_B.obs[group_name].cat.categories.tolist(), adata_B.uns[group_name + '_colors']))
        group_cmap = {**group_cmap_A, **group_cmap_B}

    G_signif = nx.DiGraph()
    G_non_signif = nx.DiGraph()
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

    # ==== Draw plot ====
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    edges_signif = G_signif.edges(data=True)
    edge_colors_signif = [node_colors[edge[0]] for edge in edges_signif]
    edge_widths_signif = [edge[2]['weight'] for edge in edges_signif]

    edges_non_signif = G_non_signif.edges(data=True)
    edge_colors_non_signif = [node_colors[edge[0]] for edge in edges_non_signif]
    edge_widths_non_signif = [edge[2]['weight'] for edge in edges_non_signif]

    for node in G_signif.nodes():
        if '_M' in node:
            nx.draw_networkx_nodes(G_signif, pos, nodelist=[node], 
                                   node_color=[node_colors[node]], 
                                   node_shape='s', 
                                   node_size=node_sizes_visual[node], 
                                   ax=ax)
        else:
            nx.draw_networkx_nodes(G_signif, pos, nodelist=[node], 
                                   node_color=[node_colors[node]], 
                                   node_size=node_sizes_visual[node], 
                                   ax=ax)

    labels = {}
    labels_pos = {}
    for cls in classes:
        labels[cls + '_L'] = cls
        labels_pos[cls + '_L'] = (pos[cls + '_L'][0]-0.2, pos[cls + '_L'][1])
    nx.draw_networkx_labels(G_signif, labels_pos, labels=labels, horizontalalignment='right', ax=ax)

    nx.draw_networkx_edges(G_signif, pos, edgelist=edges_signif, edge_color=edge_colors_signif, 
                           width=edge_widths_signif, arrowstyle='-|>', arrowsize=10, alpha=1, ax=ax)
    nx.draw_networkx_edges(G_non_signif, pos, edgelist=edges_non_signif, edge_color=edge_colors_non_signif, 
                           width=edge_widths_non_signif, arrowstyle='-|>', arrowsize=10, alpha=alpha, ax=ax)

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
    
    # ==== Save ====
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")

    return ax

def plot_MSpair_contribute_group(
    adata: anndata.AnnData,
    database_name: str = None,
    group_name: str = None,
    metabolite_name: str = None,
    summary: str = 'sender',
    cmap: str = "green",
    group_cmap: dict = None,
    figsize: tuple = (4,6),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None
):
    """
    Plot a heatmap showing group-level contributions of metabolite–sensor pairs for a specific metabolite.

    This function visualizes how each cell group contributes to the overall communication
    strength of all metabolite–sensor pairs associated with a given metabolite.
    It is particularly useful for identifying dominant sender or receiver groups for specific metabolites.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing MetaChat results.  
        Must include:
        - ``adata.uns['df_metasen_filtered']`` : table linking metabolites (HMDB IDs) to sensors.
        - ``adata.obsm['MetaChat-<database_name>-sum-<summary>']`` : communication summary matrix.
    database_name : str
        Name of the metabolite–sensor interaction database used for MCC computation.
    group_name : str
        Column name in ``adata.obs`` specifying the cell group or cluster identity.
    metabolite_name : str
        HMDB ID of the metabolite to visualize.
    summary : {"sender", "receiver"}, default="sender"
        Whether to visualize sender- or receiver-side contributions.
    cmap : {"green", "red", "blue"}, default="green"
        Colormap for the heatmap.  
        - "green": metabolic enrichment style  
        - "red": activity intensity style  
        - "blue": signal pathway style
    group_cmap : dict, optional
        Mapping from group names to colors. If ``None``, derived automatically from ``adata.uns[group_name + '_colors']``.
    figsize : tuple of float, default=(4, 6)
        Figure size (width, height).
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw the heatmap on. If ``None``, a new figure is created.
    plot_savepath : str, optional
        File path to save the figure (e.g., ``"results/MSpair_contribution_heatmap.pdf"``).  
        The format is inferred from the file extension. If ``None``, the plot is displayed interactively without saving.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the contribution heatmap.

    Notes
    -----
    - The function requires prior execution of:
        1. :func:`mc.tl.metabolic_communication`
        2. :func:`mc.tl.summary_communication`
        3. :func:`mc.tl.communication_group` or :func:`mc.tl.communication_group_spatial`
    - Each column represents a metabolite–sensor pair, and each row represents a cell group.
    - Values are log-transformed group-level summed communication scores.

    """ 

    # ==== Check inputs ====
    assert database_name is not None, "Please at least specify `database_name`."
    assert group_name is not None, "Please at least specify `group_name`."
    assert metabolite_name is not None, "Please at least specify `metabolite_name`."
    
    # ==== Prepare data ====
    df_metasen = adata.uns['df_metasen_filtered']
    name_sensor = df_metasen.loc[df_metasen['HMDB.ID'] == metabolite_name, 'Sensor.Gene'].tolist()

    if summary == 'sender':
        arrv = 's'
    elif summary == 'receiver':
        arrv = 'r'
    else:
        raise ValueError("`summary` must be either 'sender' or 'receiver'.")

    ms_pair = [f"{arrv}-{metabolite_name}-{sensor}" for sensor in name_sensor]
    ms_pair.sort()

    df_MCC = adata.obsm[f"MetaChat-{database_name}-sum-{summary}"].loc[:, ms_pair].copy()
    df_MCC[group_name] = adata.obs[group_name].copy()
    df_contribute = df_MCC.groupby(group_name).sum()
    df_contribute = np.log(df_contribute + 1)

    n_colors = 256
    if cmap == 'green':
        cmap = sns.blend_palette(["#F4FAFC", "#CAE7E0", "#80C0A5", "#48884B", "#1E4621"], n_colors=n_colors)
    elif cmap == 'blue':
        cmap = sns.blend_palette(["#FAFDFE", "#B7CDE9", "#749FD2", "#4967AC", "#3356A2"], n_colors=n_colors)
    elif cmap == 'red':
        cmap = sns.blend_palette(["#FFFEF7", "#FCF5B8", "#EBA55A", "#C23532"], n_colors=n_colors)
    else:
        raise ValueError("`cmap` must be one of {'green', 'red', 'blue'}.")

    if group_cmap is None:
        group_cmap_dict = dict(
            zip(adata.obs[group_name].cat.categories.tolist(), adata.uns[group_name + '_colors'])
        )
        group_cmap = [group_cmap_dict[g] for g in df_contribute.index]
    
    # ==== Draw plot ====
    sns.clustermap(
        df_contribute.T,
        row_cluster = False, 
        col_cluster = False, 
        col_colors = group_cmap, 
        cmap = cmap,
        figsize = figsize,
        cbar_pos=(0.02, 0.3, 0.02, 0.4)
    )

    # ==== Save ====
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")
    
    return ax

def plot_summary_pathway(
    ms_result: pd.DataFrame = None,
    metapathway_rank: pd.DataFrame = None,
    senspathway_rank: pd.DataFrame = None,
    plot_metapathway_index: list = None,
    plot_senspathway_index: list = None,
    figsize: tuple = (10,10),
    plot_savepath: str = None
):
    """
    Plot a Sankey diagram summarizing metabolic cell communication between metabolite and sensor pathways.

    This function visualizes how metabolite pathways contribute to sensor pathways based on
    group-level MCC results. Each link represents the overall communication score between
    a metabolite pathway (source) and a sensor pathway (target), scaled by interaction intensity.

    Parameters
    ----------
    ms_result : pandas.DataFrame
        A matrix of pathway-level metabolic communication scores.  
        Rows correspond to metabolite pathways, columns to sensor pathways.
        Typically generated from :func:`mc.tl.summary_pathway`.
    metapathway_rank : pandas.DataFrame
        DataFrame containing metabolite pathway rankings.
    senspathway_rank : pandas.DataFrame
        DataFrame containing sensor pathway rankings.
    plot_metapathway_index : list of int
        Indices (in ``metapathway_rank``) of metabolite pathways to include in the Sankey diagram.
    plot_senspathway_index : list of int
        Indices (in ``senspathway_rank``) of sensor pathways to include in the Sankey diagram.
    figsize : tuple of float, default=(10, 10)
        Figure size (width, height) in inches.
    plot_savepath : str, optional
        Path to save the plot as a static image (e.g., ``"results/pathway_summary.pdf"``).  
        The format is inferred from the file extension.  
        If ``None``, the figure will be displayed interactively.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Sankey diagram figure object.

    Notes
    -----
    - This function requires results computed by :func:`mc.tl.summary_pathway`.
    - Each node represents a pathway; link width corresponds to relative communication strength.
    - Scores are log-scaled and rescaled for better visual contrast.
    """

    # ==== Prepare data ====
    palette_1 = sns.color_palette("tab20",20)
    hex_colors_1 = [mcolors.to_hex(color) for color in palette_1]
    hex_colors_source = [color for index, color in enumerate(hex_colors_1) if index % 2 == 0]
    hex_colors_line = [color for index, color in enumerate(hex_colors_1) if index % 2 == 1]

    palette_2 = sns.color_palette("YlGnBu",len(plot_senspathway_index))
    hex_colors_target = [mcolors.to_hex(color) for color in palette_2][::-1]

    usename_metapathway = list(metapathway_rank.iloc[plot_metapathway_index]["Metabolite.Pathway"])
    usename_senspathway = list(senspathway_rank.iloc[plot_senspathway_index]["Sensor.Pathway"])
    ms_result_new = ms_result.loc[usename_metapathway,usename_senspathway].copy()
    
    all_values = ms_result_new.values.flatten()
    non_zero_values = all_values[all_values != 0]
    min_non_zero_value = np.min(non_zero_values)
    ms_result_new = np.log(ms_result_new/min_non_zero_value + 1)
    ms_result_new = ms_result_new.reset_index().copy()
    
    result_all_melted = ms_result_new.melt(
        id_vars='Metabolite.Pathway', var_name='Sensor.Pathway', value_name='communication_score'
    )
    
    metapathway_color = pd.DataFrame({
        'Metabolite.Pathway': np.array(ms_result_new.loc[:,"Metabolite.Pathway"]),
        'color_source': np.array(hex_colors_source[:ms_result_new.shape[0]]),
        'color_link': np.array(hex_colors_line[:ms_result_new.shape[0]])
    })
    result_all_melted = pd.merge(result_all_melted, metapathway_color, on='Metabolite.Pathway', how='outer')

    NODES = dict(
        label = usename_metapathway + usename_senspathway,
        color = hex_colors_source[:len(usename_metapathway)] + hex_colors_target[:len(usename_senspathway)]
    )

    Node_index = pd.DataFrame({
        'node': usename_metapathway + usename_senspathway,
        'index': list(range(len(usename_metapathway) + len(usename_senspathway)))
    })
    result_all_melted = pd.merge(result_all_melted, Node_index, left_on='Metabolite.Pathway', right_on = 'node', how='inner')
    result_all_melted = pd.merge(result_all_melted, Node_index, left_on='Sensor.Pathway', right_on = 'node', how='inner')

    scores = np.array(result_all_melted["communication_score"])
    scores_scaled = (scores - scores.min()) / (scores.max() - scores.min())
    scaled_width = scores_scaled * 5

    LINKS = dict(
        source = np.array(result_all_melted["index_x"]).tolist(),
        target = np.array(result_all_melted["index_y"]).tolist(),
        value = scaled_width.tolist(),
        color = np.array(result_all_melted["color_link"]).tolist()
    )

    data = go.Sankey(node = NODES, link = LINKS)
    fig = go.Figure(data)
    fig.show(config={"width": figsize[0], "height": figsize[1]})

    if plot_savepath:
        fig.write_image(plot_savepath, width=figsize[0] * 100, height=figsize[1] * 100)

def plot_metapathway_pair_contribution_bubbleplot(
    pathway_pair_contributions: dict,
    pathway_name: str,
    smallest_size: float = 10,
    cmap: str = 'blue',
    plot_title: str = None,
    figsize: tuple = (12, 5),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None
):
    """
    Plot a bubble chart showing metabolite–sensor contributions for a selected metabolic pathway.

    This function visualizes the contribution strength of metabolite–sensor pairs
    involved in a specific metabolic pathway as a bubble plot.
    Each bubble corresponds to a metabolite–sensor interaction, with color and size
    indicating the relative communication score. Missing or zero scores are displayed
    in gray bubbles with distinct outlines.

    Parameters
    ----------
    pathway_pair_contributions : dict
        This object is typically generated from :func:`mc.tl.summary_pathway`
    pathway_name : str
        The name of the metabolic pathway to visualize.
    smallest_size : float, default=10
        Base bubble size for missing (NA) or zero communication scores.
    cmap : {"blue", "green", "red"}, default="blue"
        Color gradient preset defining the color scale for communication scores.
    plot_title : str, optional
        Custom title for the figure.
    figsize : tuple of float, default=(12, 5)
        Figure size (width, height).
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw on. If ``None``, a new figure and axis are created.
    plot_savepath : str, optional
        File path to save the figure (e.g., ``"pathway_bubbleplot.pdf"``).  
        The format is inferred from the file extension. If ``None``, the plot is displayed
        interactively without saving.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the generated bubble plot.

    Notes
    -----
    - This function visualizes pathway-level MCC contributions, and is typically used
      after running pathway aggregation steps within MetaChat analysis.
    - Bubbles are categorized into quantile-based bins (≤1/3, 1/3–2/3, >2/3 quantiles)
      with increasing color intensity and bubble size.
    - Missing or zero values are shown in gray to distinguish from active interactions.
    - Legends indicate bubble categories corresponding to relative contribution scores.
    """

    # ==== Check inputs ====
    if pathway_name not in pathway_pair_contributions:
        raise ValueError(f"Pathway '{pathway_name}' not found in pathway_pair_contributions.")
    
    if cmap == 'blue':
        grad1, grad2, grad3 = '#a6cee3', '#1f78b4', '#08306b'
    elif cmap == 'green':
        grad1, grad2, grad3 = '#d9f0d3', '#66c2a4', '#238b45'
    elif cmap == 'red':
        grad1, grad2, grad3 = '#fcbba1', '#fc9272', '#de2d26'
    else:
        raise ValueError(f"Unknown cmap '{cmap}'. Choose from 'blue','green','red'.")

    # ==== Prepare data ====
    df = pathway_pair_contributions[pathway_name].copy()
    matrix = df.pivot(
        index="Metabolite.Name",
        columns="Sensor.Gene",
        values="communication_score"
    )

    row_sums = matrix.sum(axis=1)
    metabolites = row_sums.sort_values(ascending=True).index.tolist()
    sensors = matrix.columns.tolist()

    # ==== Construct plotting DataFrame ====
    data = []
    for i, met in enumerate(metabolites):
        for j, sen in enumerate(sensors):
            data.append({
                "x": j, "y": i,
                "Score": matrix.loc[met, sen],
                "Metabolite": met, "Sensor": sen
            })
    plot_df = pd.DataFrame(data)

    pos = plot_df["Score"].dropna()
    pos = pos[pos > 0]
    q1, q2 = (pos.quantile([0.33, 0.66]) if len(pos) > 0 else (0, 0))

    def style(row):
        v = row["Score"]
        if pd.isna(v):
            return {"color": "#D3D3D3", "size": smallest_size, "edge": False}
        elif v == 0:
            return {"color": "#D3D3D3", "size": smallest_size, "edge": True}
        elif v <= q1:
            return {"color": grad1,       "size": smallest_size + 20, "edge": True}
        elif v <= q2:
            return {"color": grad2,       "size": smallest_size + 40, "edge": True}
        else:
            return {"color": grad3,       "size": smallest_size + 60, "edge": True}

    styles = plot_df.apply(style, axis=1)
    plot_df["color"] = [s["color"] for s in styles]
    plot_df["size"]  = [s["size"]  for s in styles]
    plot_df["edge"]  = [s["edge"]  for s in styles]

    # ==== Draw plot ====
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for _, row in plot_df.iterrows():
        ax.scatter(
            row["x"], row["y"],
            s=row["size"],
            c=row["color"],
            edgecolors="black" if row["edge"] else row["color"],
            linewidths=0.5 if row["edge"] else 0,
            zorder=2
        )
    
    ax.set_xticks(range(len(sensors)))
    ax.set_xticklabels(sensors, rotation=45, ha="center")
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_yticks(range(len(metabolites)))
    ax.set_yticklabels(metabolites)
    ax.tick_params(axis='y', which='both', length=0)
    ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5, zorder=1)
    ax.set_xlim(-0.5, len(sensors)-0.5)
    ax.set_xlabel("Sensors")
    ax.set_ylabel("Metabolites")
    ax.set_title(plot_title or f"Metabolite–Sensor contributions in {pathway_name}")
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = [
        Line2D([], [], marker='o', linestyle='', markersize=np.sqrt(smallest_size),
               markerfacecolor='#D3D3D3', markeredgecolor='none', label='NA'),
        Line2D([], [], marker='o', linestyle='', markersize=np.sqrt(smallest_size),
               markerfacecolor='#D3D3D3', markeredgecolor='black', label='score = 0'),
        Line2D([], [], marker='o', linestyle='', markersize=np.sqrt(smallest_size + 20),
               markerfacecolor=grad1, markeredgecolor='black', label='<= 1/3 quantile'),
        Line2D([], [], marker='o', linestyle='', markersize=np.sqrt(smallest_size + 40),
               markerfacecolor=grad2, markeredgecolor='black', label='1/3–2/3 quantile'),
        Line2D([], [], marker='o', linestyle='', markersize=np.sqrt(smallest_size + 60),
               markerfacecolor=grad3, markeredgecolor='black', label='> 2/3 quantile'),
    ]
    ax.legend(
        handles=legend_elements,
        title="Score categories",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()

    # ==== Save ====
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")

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
    plot_savepath: str = None,
    return_genes: bool = False
):
    """
    Plot the smoothed gene expression profiles of metabolic cell communication response genes.

    This function visualizes the response genes inferred from metabolic cell communication analysis. 
    Genes are grouped by response clusters and displayed as a heatmap, where columns 
    correspond to genes and rows represent the order of received signals.

    The function allows highlighting top-ranked genes per cluster based on the Wald statistic 
    and sorting both genes and clusters according to their expression peak locations.

    This function should be run after :func:`mc.tl.communication_responseGenes` and :func:`mc.tl.communication_responseGenes_cluster`.

    Parameters
    ----------
    df_deg : pandas.DataFrame
        DataFrame results of response gene analysis from :func:`mc.tl.communication_responseGenes` and :func:`mc.tl.communication_responseGenes_cluster`.

    df_yhat : pandas.DataFrame
        DataFrame containing smoothed and normalized expression values of the same genes 

    show_gene_names : bool, default=True
        Whether to show gene names as x-axis tick labels in the heatmap.

    top_ngene_per_cluster : int, default=-1
        Number of top genes to display per cluster.  
        If set to a non-negative value, only the top-ranked genes (by Wald statistic) 
        within each cluster are plotted.  
        If ``-1``, all genes are included.

    colormap : str, default='magma'
        Colormap used for expression values (passed to seaborn/Matplotlib).

    cluster_colormap : str, default='Plotly'
        Qualitative colormap used to color gene clusters.  
        Accepts predefined names such as ``'Plotly'``, ``'Light24'``, ``'Dark24'``, or ``'Alphabet'``.

    color_range : tuple, optional
        Tuple ``(vmin, vmax)`` specifying the lower and upper limits for expression color scaling.  
        Values outside this range are clipped. If ``None``, full dynamic range is used.

    font_scale : float, default=1
        Scaling factor for seaborn font sizes.

    figsize : tuple of float, default=(10, 10)
        Size of the output figure (width, height).

    plot_savepath : str, optional
        File path to save the figure.  
        If ``None``, the figure is not saved.

    return_genes : bool, default=False
        Whether to return the list of genes displayed in the plot.

    Returns
    -------
    selected_genes : list of str, optional
        If ``return_genes=True``, returns a list of gene names in the order they appear 
        in the heatmap. Otherwise, returns ``None``.

    Notes
    -----
    - Clusters are first sorted by the mean position of the peak expression (``np.argmax`` of smoothed values).  
    - Within each cluster, genes are ranked by the Wald statistic, and optionally truncated 
      to the top ``N`` genes.  
    - The resulting heatmap displays genes grouped and color-coded by cluster assignment.
    - This function relies on :func:`get_cmap_qualitative` to obtain qualitative color palettes.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt

    cmap = get_cmap_qualitative(cluster_colormap)
    wald_stats = df_deg['waldStat'].values
    labels = np.array(df_deg['cluster'].values, int)
    nlabel = np.max(labels) + 1
    yhat_mat = df_yhat.values

    if color_range is not None:
        yhat_mat[yhat_mat > color_range[1]] = color_range[1]
        yhat_mat[yhat_mat < color_range[0]] = color_range[0]

    # Sort clusters by mean peak location
    peak_locs = []
    for i in range(nlabel):
        tmp_idx = np.where(labels == i)[0]
        tmp_y = yhat_mat[tmp_idx, :]
        peak_locs.append(np.mean(np.argmax(tmp_y, axis=1)))
    cluster_order = np.argsort(peak_locs)

    # Get peak per gene
    gene_peak_positions = np.argmax(yhat_mat, axis=1)

    idx = np.array([], dtype=int)
    col_colors = []

    for i in cluster_order:
        tmp_idx = np.where(labels == i)[0]

        # Step 1: sort by waldStat descending
        wald_in_cluster = wald_stats[tmp_idx]
        top_order = np.argsort(-wald_in_cluster)

        if top_ngene_per_cluster >= 0:
            top_ngene = min(len(tmp_idx), top_ngene_per_cluster)
        else:
            top_ngene = len(tmp_idx)

        selected_idx = tmp_idx[top_order[:top_ngene]]

        # Step 2: sort those by peak position
        peak_pos_selected = gene_peak_positions[selected_idx]
        final_order = np.argsort(peak_pos_selected)
        sorted_idx = selected_idx[final_order]

        idx = np.concatenate((idx, sorted_idx))
        for _ in range(len(sorted_idx)):
            col_colors.append(cmap[i % len(cmap)])

    # Plot
    sns.set(font_scale=font_scale)
    g = sns.clustermap(df_yhat.iloc[idx].T,
                       row_cluster=False,
                       col_cluster=False,
                       col_colors=col_colors,
                       cmap=colormap,
                       xticklabels=show_gene_names,
                       yticklabels=False,
                       linewidths=0,
                       figsize=figsize)
    g.ax_heatmap.invert_yaxis()
    g.cax.set_position([.1, .2, .03, .45])

    if plot_savepath is not None:
        plt.savefig(plot_savepath, dpi=300)

    if return_genes:
        return list(df_deg.iloc[idx].index)


def plot_communication_responseGenes_keggEnrich(
    df_result: pd.DataFrame,
    organism: str = "Human",
    show_term_order: list = [0,1,2,3,4],
    cmap: str = 'green',
    maxshow_gene: int = 5,
    figsize: tuple = (6,6),
    ax: Optional[mpl.axes.Axes] = None,  
    plot_savepath: str = None
):
    """
    Plot a horizontal bar chart summarizing KEGG enrichment results of MCC response genes.

    This function visualizes the top enriched KEGG pathways identified from
    :func:`mc.tl.communication_responseGenes`, with bar length representing 
    -log10(p-value) and text annotations showing top associated genes.

    Parameters
    ----------
    df_result : pandas.DataFrame
        The KEGG enrichment results table returned by :func:`mc.tl.communication_responseGenes`.
    organism : {"Human", "Mouse"}, default="Human"
        If set to ``"Mouse"``, gene names are capitalized (first letter uppercase, rest lowercase).
    show_term_order : list of int, default=[0,1,2,3,4]
        List of row indices (in ``df_result``) specifying which pathways to show and their order.
        If ``None``, the top 5 terms will be displayed.
    cmap : {"green", "blue", "red"}, default="green"
        Color theme for the barplot.
    maxshow_gene : int, default=10
        Maximum number of genes to display under each pathway.
    figsize : tuple of float, default=(6, 6)
        Figure size (width, height).
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw the barplot on. If ``None``, a new figure and axis are created.
    plot_savepath : str, optional
        File path to save the figure (e.g., ``"results/kegg_enrich.pdf"``).  
        The format is inferred from the file extension. If ``None``, the plot is shown interactively.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the bar chart.

    Notes
    -----
    - This visualization summarizes KEGG enrichment performed by
      :func:`mc.tl.communication_responseGenes`.
    - Each bar represents one KEGG pathway, with its significance shown as -log10(p-value).
    - Up to ``maxshow_gene`` top genes per pathway are displayed below the bar label.
    """
    
    # ==== Prepare data ====
    df_show = df_result.iloc[show_term_order].copy()
    x_names = df_show['Term'].tolist()
    x_names.reverse()

    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}
    path_to_genes = {}
    path_to_value = {}

    for _, row in df_show.iterrows():
        gene_list = row['Genes'].split(';')
        # Capitalize gene names if organism is Mouse
        if organism.lower() == "mouse":
            gene_list = [g.capitalize() for g in gene_list]
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = [x_to_num[v] for v in x_names]
    y = [path_to_value[v] for v in x_names]
    ax.barh(x, y, color=bar_color[cmap], height=0.5) 
    ax.set_facecolor('white')

    for v in x_names:
        ax.text(0 + 0.05, x_to_num[v], v, color='black', ha='left', va='center')
        ax.text(0 + 0.05, x_to_num[v] - 0.4, path_to_genes[v], color=text_color[cmap], ha='left', va='center')

    ax.set_yticklabels([])
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_ylim([-0.7, max(x) + 1])
    ax.set_xlabel('-log10(p-value)')
    ax.set_ylabel('KEGG pathway')

    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")

    return ax

def plot_DEG_volcano(
    deg_result: pd.DataFrame,
    name_col: str = "names",
    logfc_col: str = "logfoldchanges",
    pval_col: str = "pvals_adj",
    logfc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    neglog10_cap: float = 50.0,
    pointsize_min: float = 5.0,
    pointsize_max: float = 30.0,
    label_thresh_neglog10: float = 10.0,
    label_fontsize: int = 6,
    figsize: tuple = (5, 5),
    cbar_orientation: str = "horizontal",
    cbar_pad: float = 0.2,
    title: str = "Volcano Plot",
    show_labels: bool = True,
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None
):
    """
    Plot a volcano plot from differential MCC results.

    This function visualizes log2 fold changes versus statistical significance
    (-log10 adjusted p-values) to highlight up- and down-regulated genes.

    Parameters
    ----------
    deg_result : pandas.DataFrame
        DataFrame containing differential MCC results.
    name_col : str, default='names'
        Column name for gene or feature names.
    logfc_col : str, default='logfoldchanges'
        Column name for log2 fold changes.
    pval_col : str, default='pvals_adj'
        Column name for adjusted p-values.
    logfc_thresh : float, default=1.0
        Threshold for absolute log2 fold change used to draw vertical reference lines.
    pval_thresh : float, default=0.05
        Threshold for adjusted p-value used to draw a horizontal reference line.
    neglog10_cap : float, default=50.0
        Cap for -log10(p-value) to avoid extreme values distorting visualization.
    pointsize_min : float, default=5.0
        Minimum scatter point size.
    pointsize_max : float, default=30.0
        Maximum scatter point size.
    label_thresh_neglog10 : float, default=10.0
        Threshold of -log10(p-value) above which gene labels are annotated.
    label_fontsize : int, default=6
        Font size for gene labels.
    figsize : tuple of float, default=(5, 5)
        Figure size (width, height) in inches.
    cbar_orientation : {'horizontal', 'vertical'}, default='horizontal'
        Orientation of the colorbar.
    cbar_pad : float, default=0.2
        Padding distance between colorbar and axis.
    title : str, default='Volcano Plot'
        Title of the figure.
    show_labels : bool, default=True
        Whether to show text labels for significant genes.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw on.  
        If None, a new figure and axis are created.
    plot_savepath : str, optional
        File path to save the figure (e.g. ``'results/volcano_plot.pdf'``).  
        The format is inferred from the file extension.  
        If None, the figure is shown interactively.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis object containing the rendered volcano plot.

    Notes
    -----
    - Points are colored by log2 fold change using a diverging colormap.
    - Vertical and horizontal dashed lines represent significance thresholds.
    - Extreme -log10(p-values) are capped at ``neglog10_cap`` to maintain scale.
    """
    
    # ==== Copy and preprocess ====
    deg_result = deg_result.copy()
    deg_result['neg_log10_pValue'] = -np.log10(deg_result[pval_col])
    deg_result['neg_log10_pValue'] = np.clip(deg_result['neg_log10_pValue'], None, neglog10_cap)

    # ==== Point size and color normalization ====
    deg_result['point_size'] = np.clip(deg_result['neg_log10_pValue'], pointsize_min, pointsize_max)
    vmin = deg_result[logfc_col].min()
    vmax = deg_result[logfc_col].max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # ==== Initialize figure ====
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # ==== Draw scatter ====
    scatter = ax.scatter(
        x=deg_result[logfc_col],
        y=deg_result['neg_log10_pValue'],
        c=deg_result[logfc_col],
        s=5 * deg_result['point_size'],
        cmap='Spectral_r',
        norm=norm
    )

    # ==== Add colorbar ====
    cbar = plt.colorbar(scatter, ax=ax, orientation=cbar_orientation, pad=cbar_pad, shrink=0.6)
    cbar.set_label('Log2 Fold Change')

    # ==== Axis formatting ====
    max_fc = np.max(np.abs(deg_result[logfc_col])) + 0.5
    ax.set_xlim(-max_fc, max_fc)
    ax.set_ylim(-3, neglog10_cap + 5)
    ax.set_title(title)
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 p-value')

    # Add threshold lines
    ax.axhline(y=-np.log10(pval_thresh), color='black', linestyle='--')
    ax.axvline(x=logfc_thresh, color='black', linestyle='--')
    ax.axvline(x=-logfc_thresh, color='black', linestyle='--')

    # Annotate significant points with adjustText
    if show_labels:
        sig_df = deg_result[
            (np.abs(deg_result[logfc_col]) > logfc_thresh) &
            (deg_result['neg_log10_pValue'] > label_thresh_neglog10)
        ]

        texts = []
        for _, row in sig_df.iterrows():
            texts.append(
                ax.text(
                    row[logfc_col],
                    row['neg_log10_pValue'],
                    str(row[name_col]),
                    fontsize=label_fontsize
                )
            )
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.set_box_aspect(1)
    plt.tight_layout()

    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")

    return ax

def plot_3d_feature(
    adata: anndata.AnnData,
    feature: str,
    spatial_key: str = "spatial_3d",
    cmap_continuous: str = "Viridis",
    layer: str = None,
    use_raw: bool = False,
    point_size: int = 3,
    opacity = 0.6,
    aspectratio: tuple = (1.1, 1.0, 1.3),
    show_axes: bool = True,
    figsize: tuple = (9, 9),
    plot_savepath: str = None
):
    """
    Visualize a spatial feature (gene or obs annotation) in 3D using Plotly.

    This function renders a 3D scatter plot for any feature stored in `.obs`, 
    `.var`, `.layers`, or `.raw`, using the spatial coordinates stored in 
    ``adata.obsm[spatial_key]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with 3D spatial coordinates stored in ``adata.obsm[spatial_key]``.
    feature : str
        Name of the feature to visualize. It can be:
        - A column name in ``adata.obs`` (categorical or continuous);
        - A gene name in ``adata.var_names`` or ``adata.raw.var_names``.
    spatial_key : str, default='spatial_3d'
        Key in ``adata.obsm`` containing the 3D spatial coordinates (shape: ``(n_obs, 3)``).
    cmap_continuous : str, default='Viridis'
        Colormap for continuous features.
    layer : str, optional
        If provided, use expression values from ``adata.layers[layer]``.
    use_raw : bool, default=False
        Whether to use ``adata.raw`` instead of ``adata.X`` when available.
    point_size : int, default=3
        Size of scatter plot markers.
    opacity : float or list, default=0.6
        Global opacity (float) or per-category opacity (list, same length as categories).
    aspectratio : tuple of float, default=(1.1, 1.0, 1.3)
        Aspect ratio (x, y, z) for 3D axes scaling.
    show_axes : bool, default=True
        Whether to display 3D axis lines and labels.
    figsize : tuple of float, default=(9, 9)
        Figure size in inches (controls pixel width/height of the Plotly figure).
    plot_savepath : str, optional
        Path to save the 3D figure (e.g. ``"results/3D_feature_geneA.pdf"``).  
        If None, the figure is displayed interactively.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated Plotly 3D scatter figure.

    Notes
    -----
    - If the feature is categorical, it is colored using discrete color sequences.
    - If the feature is continuous (e.g., gene expression), a continuous colormap is applied.
    - This function is ideal for visualizing 3D tissue reconstructions, such as those derived
      from MetaChat’s 3D MCC analysis.
    """

    if spatial_key not in adata.obsm:
        raise KeyError(f"`obsm['{spatial_key}']` not found or not set to 3D coords.")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"`obsm['{spatial_key}']` must be of shape (n_cells, 3).")

    n = coords.shape[0]
    df = adata.obs.copy()
    df["x_3d"], df["y_3d"], df["z_3d"] = coords[:, 0], coords[:, 1], coords[:, 2]

    def _to_1d_array(x):
        if sparse.issparse(x):
            x = x.toarray()
        x = np.asarray(x)
        if x.ndim == 2:
            if x.shape[0] == 1:
                x = x[0]
            elif x.shape[1] == 1:
                x = x[:, 0]
        return x.reshape(-1)

    is_obs_feature = feature in adata.obs.columns
    gene_vec = None
    if not is_obs_feature:
        if use_raw and getattr(adata, "raw", None) is not None and feature in adata.raw.var_names:
            X = adata.raw[:, feature].X
            gene_vec = _to_1d_array(X)
        elif feature in adata.var_names:
            if layer is not None:
                if layer not in adata.layers:
                    raise KeyError(f"Layer '{layer}' not found in `adata.layers`.")
                X = adata[:, feature].layers[layer]
            else:
                X = adata[:, feature].X
            gene_vec = _to_1d_array(X)
        if gene_vec is not None and gene_vec.shape[0] != n:
            raise ValueError(f"Gene vector length ({gene_vec.shape[0]}) does not match number of cells ({n}).")

    # --- main plotting ---
    if is_obs_feature:
        df["feature"] = adata.obs[feature]
        is_categorical = pd.api.types.is_categorical_dtype(df["feature"]) or df["feature"].dtype == "category"

        if is_categorical:
            color_key = feature + "_colors"
            colors = adata.uns.get(color_key, None)
            category_order = list(df["feature"].cat.categories) if hasattr(df["feature"], "cat") else sorted(df["feature"].unique())

            fig = px.scatter_3d(
                df, x="x_3d", y="y_3d", z="z_3d",
                color="feature",
                color_discrete_sequence=colors if colors is not None else None,
                category_orders={"feature": category_order},
                title=f"3D Spatial - obs: {feature}",
            )
            # --- opacity by category ---
            if isinstance(opacity, (list, tuple)) and len(opacity) >= len(category_order):
                for i, trace in enumerate(fig.data):
                    trace.opacity = opacity[i]
            else:
                for trace in fig.data:
                    trace.opacity = opacity if isinstance(opacity, (int, float)) else 0.6

        else:
            try:
                df["feature"] = pd.to_numeric(df["feature"])
            except Exception:
                pass
            fig = px.scatter_3d(
                df, x="x_3d", y="y_3d", z="z_3d",
                color="feature",
                color_continuous_scale=cmap_continuous,
                title=f"3D Spatial - obs: {feature}",
            )
            for trace in fig.data:
                trace.opacity = opacity if isinstance(opacity, (int, float)) else 0.6

    else:
        if gene_vec is None:
            candidates = []
            if getattr(adata, "raw", None) is not None:
                candidates.append("adata.raw.var_names")
            candidates.append("adata.var_names")
            raise KeyError(f"Feature '{feature}' not found in `adata.obs` or as a gene in {', '.join(candidates)}.")
        df["feature"] = gene_vec
        fig = px.scatter_3d(
            df, x="x_3d", y="y_3d", z="z_3d",
            color="feature",
            color_continuous_scale="Viridis",
            title=f"3D Spatial - gene: {feature}",
        )
        for trace in fig.data:
            trace.opacity = opacity if isinstance(opacity, (int, float)) else 0.6

    fig.update_traces(marker_size=point_size)
    fig.update_layout(
        height=figsize[0]*100,
        width=figsize[1]*100,
        scene=dict(
            aspectratio=dict(
                x=float(aspectratio[0]),
                y=float(aspectratio[1]),
                z=float(aspectratio[2])
            ),
            xaxis=dict(title="X" if show_axes else "", showbackground=False, showgrid=False,
                       zeroline=show_axes, showline=show_axes,
                       linecolor="black", linewidth=3 if show_axes else 0,
                       showticklabels=show_axes),
            yaxis=dict(title="Y" if show_axes else "", showbackground=False, showgrid=False,
                       zeroline=show_axes, showline=show_axes,
                       linecolor="black", linewidth=3 if show_axes else 0,
                       showticklabels=show_axes),
            zaxis=dict(title="Z" if show_axes else "", showbackground=False, showgrid=False,
                       zeroline=show_axes, showline=show_axes,
                       linecolor="black", linewidth=3 if show_axes else 0,
                       showticklabels=show_axes),
        ),
        margin=dict(r=0, l=0, b=0, t=30),
    )
    fig.show()

    if plot_savepath:
        fig.write_image(plot_savepath, width=figsize[0]*100, height=figsize[1]*100)

def plot_dis_thr(
    adata: anndata.AnnData,
    dis_thr: float,
    spot_index: int,
    use_existing_distance: bool = False,
    figsize: tuple = (6, 5),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None,
):
    """
    Visualize spatial neighborhood within a specified distance threshold around a selected spot.

    This function highlights which spots in the spatial omics dataset fall within a given Euclidean distance (`dis_thr`) from a selected spot.  
    The center spot is shown in a distinct color, while neighboring and non-neighboring spots are visually separated.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing spatial coordinates in ``adata.obsm["spatial"]``.
    dis_thr : float
        Distance threshold (in pixel or micron units, depending on ``adata`` scaling) used to define spatial neighbors.
    spot_index : int
        Index (0-based) of the reference/center spot.
    use_existing_distance : bool, default=False
        Whether to reuse an existing precomputed distance matrix stored in
        ``adata.obsp["spatial_distance"]``.  
        If False, a new Euclidean distance matrix will be computed.
    figsize : tuple of float, default=(6, 5)
        Figure size (width, height) in inches.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw on.  
        If None, a new figure and axis are created.
    plot_savepath : str, optional
        File path to save the plot (e.g., ``"results/dis_thr_spots.pdf"``).  
        The format is inferred from the file extension.  
        If None, the plot is shown interactively.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the rendered spatial scatter plot.

    Notes
    -----
    - This function helps visualize local spot neighborhoods used in spatial communication
      or proximity-based analyses (e.g., for barrier-aware OT or diffusion modeling).
    - The color coding is:
        * **center** – the selected reference spot  
        * **neighbor** – spots within `dis_thr` distance  
        * **outside** – all other spots
    """
    adata_vis = adata.copy()

    # compute or reuse distance matrix
    if use_existing_distance and 'spatial_distance' in adata_vis.obsp:
        dis_mat = adata_vis.obsp['spatial_distance']
    else:
        dis_mat = distance_matrix(adata_vis.obsm["spatial"], adata_vis.obsm["spatial"])
        adata_vis.obsp['spatial_distance'] = dis_mat

    # compute mask: 0 = outside, 1 = within dis_thr, 2 = center
    mask = (dis_mat[spot_index, :] < dis_thr).astype(int)
    spot_name = adata_vis.obs.index[spot_index]
    mask_series = pd.Series(mask, index=adata_vis.obs.index)
    mask_series.loc[spot_name] = 2

    # map numeric to string labels
    label_map = {0: 'outside', 1: 'neighbor', 2: 'center'}
    label_series = mask_series.map(label_map).astype('category')

    # assign to obs
    adata_vis.obs['within_dis_thr'] = label_series
    
    fig, ax = plt.subplots(figsize = figsize)
    sq.pl.spatial_scatter(
        adata_vis,
        color='within_dis_thr',
        title=f'Spots within dis_thr={dis_thr} of spot {spot_index}',
        ax = ax
    )
    ax.set_box_aspect(1)

    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")
    
    return ax

def plot_LRC_markers(
    adata: anndata.AnnData,
    LRC_name: str,
    LRC_marker_genes: list, 
    avg: bool = False,
    figsize = (10, 5),
    plot_savepath: str = None
):
    """
    Visualize expression of LRC (Long-Range Channel) marker genes in spatial omics data.

    This function plots the spatial expression of selected LRC marker genes.  
    It can either visualize individual genes separately or compute and plot the
    average expression of all selected markers.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with spatial coordinates stored in ``adata.obsm["spatial"]``.
    LRC_name : str
        Name tag for the LRC type (e.g., "CSF", "Blood").
        Used in plot titles and output naming.
    LRC_marker_genes : list
        List of LRC marker gene names to visualize.
        Genes not found in ``adata.var_names`` are automatically skipped.
    avg : bool, default=False
        If True, plots the average expression of all marker genes.  
        If False, plots each gene separately.
    figsize : tuple of float, default=(10, 5)
        Figure size (width, height) in inches.
    plot_savepath : str, optional
        Path to save the figure (e.g. ``"results/LRC_markers_Blood.pdf"``).  
        If None, the figure is displayed interactively.

    Returns
    -------
    ax : matplotlib.axes.Axes or list of Axes
        Matplotlib axis or list of axes containing the rendered plot(s).

    Notes
    -----
    - Marker genes are filtered to retain only those present in ``adata.var_names``.
    - Average expression is computed directly from ``adata.X`` (converted to dense if needed).
    - When multiple genes are plotted, they are arranged in a single row of subplots.

    """
    
    # Filter out genes not present
    valid_genes = [g for g in LRC_marker_genes if g in adata.var_names]
    if len(valid_genes) == 0:
        raise ValueError("None of the input marker genes were found in adata.var_names.")
    
    if avg:
        expr = adata[:, valid_genes].X
        if not isinstance(expr, np.ndarray):
            expr = expr.toarray()
        avg_expr = expr.mean(axis=1)
        obs_key = f"LRC_{LRC_name}_avg_markers"
        adata.obs[obs_key] = avg_expr

        fig, ax = plt.subplots(figsize=figsize)
        sq.pl.spatial_scatter(adata, color=obs_key, ax=ax)
        ax.set_title(obs_key)
        ax.set_box_aspect(1)
        plt.tight_layout()

    else:
        n_genes = len(valid_genes)
        fig, axes = plt.subplots(1, n_genes, figsize=(6 * n_genes, 5))
        if n_genes == 1:
            axes = [axes]
        
        for gene, ax in zip(valid_genes, axes):
            obs_key = f"LRC_{LRC_name}_marker_{gene}"
            adata.obs[obs_key] = adata[:, gene].X.toarray().ravel()
            sq.pl.spatial_scatter(adata, color=gene, ax=ax)
            ax.set_title(f"{gene}")
            ax.set_box_aspect(1)
        plt.suptitle(f"LRC markers for {LRC_name}", fontsize=14)
        plt.tight_layout()
    
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")

    return ax

def plot_spot_distance(
    adata: anndata.AnnData,
    dist_matrix_key: str,
    spot_index: int,
    figsize: tuple = (6, 5),
    ax: Optional[mpl.axes.Axes] = None,
    plot_savepath: str = None
):
    """
    Visualize the spatial distance from a selected spot to all other spots.

    This function displays the distance values (from a specified spot) stored
    in a precomputed distance matrix under ``adata.obsp``.  
    It helps assess local or long-range connectivity patterns in spatial data.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing spatial coordinates in
        ``adata.obsm["spatial"]`` and a distance matrix stored in ``adata.obsp``.
    dist_matrix_key : str
        Key name in ``adata.obsp`` where the spatial distance matrix is stored,
        e.g., ``"spatial_distance"`` or ``"spatial_distance_LRC_base"``.
    spot_index : int
        Index of the target spot (0-based) whose distances to all other spots
        will be visualized.
    figsize : tuple of float, default=(6, 5)
        Figure size (width, height) in inches.
    plot_savepath : str, optional
        Path to save the plot (e.g. ``"results/distance_to_spot120.pdf"``).  
        The format is inferred from the file extension.  
        If ``None``, the figure is shown interactively.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw the plot on.  
        If ``None``, a new figure and axis will be created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the rendered spatial distance visualization.

    Notes
    -----
    - Distances are taken directly from the precomputed matrix in ``adata.obsp``.
    - The distances are stored in ``adata.obs["distance_from_target"]`` for plotting.
    - The selected target spot (by index) can be visually identified as having
      the minimum distance (0).
    """
    adata_vis = adata.copy()

    # ==== Extract distance vector ====
    if dist_matrix_key not in adata_vis.obsp:
        raise KeyError(f"Distance matrix '{dist_matrix_key}' not found in adata.obsp.")
    dist_matrix = adata.obsp[dist_matrix_key]
    distances_from_target = dist_matrix[spot_index]
    adata_vis.obs["distances_from_target"] = distances_from_target

    # ==== Plot ====
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sq.pl.spatial_scatter(
        adata_vis, 
        color="distances_from_target", 
        ax=ax,
        title=f"Distance to spot index {spot_index}"
    )

    ax.set_box_aspect(1)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    # ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])  # Flip y-axis
    plt.tight_layout()
    plt.show()

    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")

    return ax
    
def plot_graph_connectivity(
    G: nx.Graph,
    node_size: float = 10.0,
    linewidths: float = 1.0,
    width: float = 1.0,
    show_weights: bool = False,
    weight_fontsize: int = 8,
    figsize: tuple = (8, 8),
    plot_savepath: str = None
):
    """
    Plot a 2D visualization of a graph showing connectivity between nodes and edges.

    This function visualizes a graph structure (e.g., communication connectivity or
    spatial relationships) using node coordinates as (x, y) positions.  
    Optionally, edge weights can be displayed as labels.

    Parameters
    ----------
    G : networkx.Graph
        Input graph where each node represents a 2D point (tuple of x, y coordinates)
        and edges may have an optional ``'weight'`` attribute.
    node_size : float, default=10.0
        Marker size of nodes.
    linewidths : float, default=1.0
        Line width of node borders.
    width : float, default=1.0
        Width of edges.
    show_weights : bool, default=False
        Whether to display numerical edge weights on the plot.
    weight_fontsize : int, default=8
        Font size of edge weight annotations (effective only if ``show_weights=True``).
    figsize : tuple of float, default=(8, 8)
        Figure size (width, height) in inches.
    plot_savepath : str, optional
        File path to save the figure (e.g., ``"results/graph_connectivity.png"``).  
        If ``None``, the figure is shown interactively.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axis containing the rendered graph.

    Notes
    -----
    - Node coordinates are assumed to be stored as the node keys, e.g. ``(x, y)`` tuples.
    - Edge weights (if any) are visualized with labels when ``show_weights=True``.
    - Commonly used for inspecting graph connectivity matrices derived from spatial MCC data.
    """

    # ==== Prepare data ====
    pos = {node: node for node in G.nodes()}
      
    plt.figure(figsize=figsize)
    nx.draw(
        G, 
        pos, 
        node_size=node_size, 
        edge_color="gray", 
        alpha=0.5,
        linewidths=linewidths, 
        width=width
    )

    # Display edge weights if enabled
    if show_weights:
        edge_labels = {
            (u, v): f"{data['weight']:.2f}" 
            for u, v, data in G.edges(data=True) 
            if 'weight' in data
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=weight_fontsize
        )

    plt.title("Graph Connectivity with Edge Weights" if show_weights else "Graph Connectivity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    if plot_savepath:
        plt.savefig(plot_savepath, dpi=300, bbox_inches="tight")

def plot_direction_similarity(
    df_direction: pd.DataFrame,
    cluster_labels: np.ndarray,
    cmap="mako",
    figsize=(9, 7),
    title="Flow pattern similarity (Euclidean-based)",
    savepath=None
):
    """
    Plot a block-ordered similarity heatmap for direction-based flow clusters.

    This function visualizes the pairwise similarity of MCC flow direction histograms
    across all metabolite–sensor pairs, ordered by K-means cluster labels.
    The resulting heatmap highlights within-cluster consistency and
    between-cluster differences in flow directionality patterns.

    The similarity between two pairs is defined as:
        ``S = 1 - D / D.max()``,
    where ``D`` is the pairwise Euclidean distance between direction histograms.
    Thus, S ranges from 0 (completely dissimilar) to 1 (identical).

    Parameters
    ----------
    df_direction : pandas.DataFrame
        Matrix of direction histogram features (rows = metabolite–sensor pairs,
        columns = direction bins, typically 18 bins).
    cluster_labels : numpy.ndarray or array-like
        Cluster assignments for each M–S pair, usually obtained from K-means or
        hierarchical clustering.
    cmap : str or matplotlib Colormap, default="mako"
        Colormap for the similarity heatmap. Can be any seaborn or matplotlib colormap.
    figsize : tuple of (float, float), default=(9, 7)
        Figure size in inches.
    title : str, default="Flow pattern similarity (Euclidean-based)"
        Title displayed at the top of the plot.
    savepath : str or None, optional
        If provided, saves the figure to the specified file path.

    Returns
    -------
    None
        Displays a block-ordered heatmap showing inter-pair similarity
        and cluster boundaries.

    Notes
    -----
    - The similarity matrix is computed as ``1 - D / D.max()``,
      where D is the Euclidean distance matrix among all histograms.
    - The function automatically sorts rows and columns by cluster label
      to create block-like patterns for visual cluster inspection.
    - Colored sidebars indicate cluster membership along both axes.
    - A legend summarizes each flow pattern cluster with its sample size.
    """
    
    # ---------- Compute similarity ----------
    X = df_direction.values.astype(float)
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X, metric="euclidean")
    S = 1 - D / D.max()
    np.fill_diagonal(S, 1.0)

    # ---------- Order by cluster ----------
    order = np.argsort(cluster_labels)
    S_ord = S[order][:, order]
    labs_ord = cluster_labels[order]
    uniq = np.unique(labs_ord)

    # ---------- Cluster color mapping ----------
    palette = list(mcolors.TABLEAU_COLORS.values())
    cmap_by_cluster = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}

    # ---------- Cluster bounds ----------
    bounds, sizes = [], {}
    start = 0
    for c in uniq:
        k = int((labs_ord == c).sum())
        sizes[c] = k
        bounds.append((c, start, start + k))
        start += k

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        S_ord,
        vmin=np.percentile(S, 2),
        vmax=np.percentile(S, 98),
        cmap = cmap,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Similarity (1 - normalized Euclidean distance)"}
    )

    # White grid lines
    n = S_ord.shape[0]
    ax.plot(np.arange(n) + 0.5, np.arange(n) + 0.5, color="white", lw=0.5, zorder=5)
    for _, s, e in bounds:
        ax.axhline(s, color="white", lw=0.6)
        ax.axhline(e, color="white", lw=0.6)
        ax.axvline(s, color="white", lw=0.6)
        ax.axvline(e, color="white", lw=0.6)

    # Sidebars
    bar_width = 3.0
    for c, s, e in bounds:
        ax.add_patch(patches.Rectangle((-bar_width, s), bar_width, (e - s),
                        facecolor=cmap_by_cluster[c], edgecolor="none",
                        transform=ax.transData, clip_on=False, zorder=10))
        ax.add_patch(patches.Rectangle((s, -bar_width), (e - s), bar_width,
                        facecolor=cmap_by_cluster[c], edgecolor="none",
                        transform=ax.transData, clip_on=False, zorder=10))

    # Legend
    handles = [
        mlines.Line2D([], [], color=cmap_by_cluster[c], marker="s", linestyle="None",
                      markersize=8, label=f"Pattern {c} (n={sizes[c]})")
        for c in uniq
    ]
    ax.legend(
        handles=handles,
        title="Flow patterns",
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.2, 1.0),
        borderaxespad=0.0
    )

    ax.set_title(title)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_3d_LRC_with_two_slices(
    adata,
    mask_key: str,
    spatial_key: str = "spatial_3d",
    z_levels: list = None,      # e.g., [z1, z2]; if None, auto quantiles (0.3, 0.7)
    slab_ratio: float = 0.02,   # thinner slab so it's cleaner
    flatten_z: bool = True,     # flatten points in slice to exactly z0
    bg_color: str = "#b0b0b0",
    ch_color: str = "#ff7f0e",
    plane_opacity: float = 0.35,
):
    """
    Visualize 3D long-range channels (LRCs) with two representative z-slice views.

    This function creates a 3D interactive Plotly visualization showing the overall
    distribution of long-range channels (LRCs) and two sectional slices at selected
    z-levels. It helps interpret the spatial continuity of LRCs across depth and
    verify their anatomical localization relative to tissue layers.

    The figure layout consists of:
      1. **Left panel** — full 3D overview of the tissue with highlighted LRC voxels.
      2. **Middle and right panels** — top-down slice views at two z-planes,
         showing LRCs within thin slabs (controlled by `slab_ratio`).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing 3D spatial coordinates in ``.obsm[spatial_key]``
        and a binary mask column in ``.obs[mask_key]`` indicating LRC membership.
    mask_key : str
        Key in ``adata.obs`` corresponding to a boolean array marking LRC points
        (e.g., ``'LRC_CSF_manual_filtered'`` or ``'LRC_Blood_auto'``).
    spatial_key : str, default="spatial_3d"
        Key in ``adata.obsm`` specifying 3D spatial coordinates (shape: N × 3).
    z_levels : list of float, optional
        Two z-levels to use for slice visualization.
        If ``None``, the function automatically selects the 0.3 and 0.7 quantiles
        of the z-coordinate distribution.
    slab_ratio : float, default=0.02
        Fraction of the z-range defining the slab thickness for each slice.
        Smaller values yield thinner and cleaner cross-sections.
    flatten_z : bool, default=True
        If True, all points within a slice are flattened to the central z-plane
        (for a 2D-like appearance). If False, preserves their original depth variation.
    bg_color : str, default="#b0b0b0"
        Color for non-LRC (background) points.
    ch_color : str, default="#ff7f0e"
        Color for highlighted LRC points (channels).
    plane_opacity : float, default=0.35
        Opacity of the grey slicing planes shown in the main 3D view.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly figure object containing three synchronized 3D scenes:
        the main tissue view and two z-slice panels.

    Notes
    -----
    - The function uses Plotly's ``make_subplots`` with three 3D scenes arranged
      horizontally for combined visualization.
    - The selected z-levels are marked by semi-transparent planes in the main panel.
    - Each slice subplot is rendered with an orthographic top-down projection
      to emphasize spatial distribution patterns rather than depth.
    - This function is useful for verifying whether manually or automatically
      identified LRC regions form continuous 3D paths through the tissue volume.
    """
    
    # --- prepare data ---
    coords = np.asarray(adata.obsm[spatial_key])
    obs = adata.obs
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    m = obs[mask_key].astype(bool).to_numpy()

    zmin, zmax = float(z.min()), float(z.max())
    if z_levels is None:
        z_levels = list(np.quantile(z, [0.3, 0.7]))
    thick = (zmax - zmin) * slab_ratio

    # --- layout: 1 row (main + two slices) ---
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type":"scene"},{"type":"scene"},{"type":"scene"}]],
        column_widths=[0.5, 0.25, 0.25],
        subplot_titles=("LRC overview",
                        f"Slice z≈{z_levels[0]:.2f}",
                        f"Slice z≈{z_levels[1]:.2f}")
    )

    # ===== Main plot =====
    # background
    fig.add_trace(go.Scatter3d(
        x=x[~m], y=y[~m], z=z[~m],
        mode="markers",
        marker=dict(size=1, color=bg_color, opacity=0.1),
        name="background"
    ), row=1, col=1)
    # channel
    fig.add_trace(go.Scatter3d(
        x=x[m], y=y[m], z=z[m],
        mode="markers",
        marker=dict(size=3, color=ch_color, opacity=0.7),
        name="channel"
    ), row=1, col=1)
    # planes
    xx = np.linspace(x.min(), x.max(), 2)
    yy = np.linspace(y.min(), y.max(), 2)
    X, Y = np.meshgrid(xx, yy)
    for z0 in z_levels:
        Z = np.full_like(X, z0)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            showscale=False, opacity=plane_opacity,
            surfacecolor=np.zeros_like(X),
            colorscale=[[0,"grey"],[1,"grey"]],
            hoverinfo="skip"
        ), row=1, col=1)

    # ===== helper: add one slice subplot =====
    def add_slice(col, z0):
        sel = (z >= z0 - thick/2) & (z <= z0 + thick/2)
        xb, yb, zb = x[sel & ~m], y[sel & ~m], z[sel & ~m]
        xc, yc, zc = x[sel &  m], y[sel &  m], z[sel &  m]
        if flatten_z:
            zb = np.full_like(zb, z0)  # flatten to one layer
            zc = np.full_like(zc, z0)

        # background (faint)
        fig.add_trace(go.Scatter3d(
            x=xb, y=yb, z=zb,
            mode="markers",
            marker=dict(size=2.5, color=bg_color, opacity=0.1),
            showlegend=False
        ), row=1, col=col)
        # channel (highlight)
        fig.add_trace(go.Scatter3d(
            x=xc, y=yc, z=zc,
            mode="markers",
            marker=dict(size=3, color=ch_color, opacity=0.9),
            showlegend=False
        ), row=1, col=col)

        # top-down orthographic view to avoid multi-layer look
        fig.update_layout(**{
            f"scene{col}": dict(
                camera=dict(projection=dict(type="orthographic"),
                            eye=dict(x=0., y=0., z=2.0), up=dict(x=0, y=1, z=0)),
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            )
        })

    add_slice(2, z_levels[0])
    add_slice(3, z_levels[1])

    # main scene style
    fig.update_layout(
        scene=dict(
            camera=dict(projection=dict(type="orthographic")),
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
        ),
        margin=dict(l=0,r=0,t=40,b=0),
        paper_bgcolor="white"
    )
    return fig