# ============================================================
import numpy as np
import pandas as pd

from pydpc import Cluster

import matplotlib as mpl
import matplotlib.pyplot as plt

import anndata
# ============================================================

def LRC_unfiltered(
    adata: anndata.AnnData,
    LRC_name: str = None,
    LRC_source: str = "marker",
    obs_name: str = None,  
    quantile: float = 90.0,
    copy: bool = False
):
    """
    Identify unfiltered candidate LRC (long-range channel) spots based on the quantile of a marker feature.

    This function selects candidate points whose marker feature (e.g., gene expression or score)
    exceeds a specified quantile threshold. The result is stored in
    ``adata.obs['LRC_<LRC_name>_<LRC_source>_unfiltered']`` as categorical values (0 or 1).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with shape ``n_obs × n_var``.
    LRC_name : str
        The name of the long-range channel (e.g., ``'Blood'`` or ``'CSF'``).
    LRC_source : str, default='marker'
        The type of feature used for selection (e.g., ``'marker'``, ``'score'``).
        This will be included in the generated column name.
    obs_name : str
        The key in ``adata.obs`` containing the numeric feature used for quantile selection.
    quantile : float, default=90.0
        The percentile threshold (0–100).  
        Example: 90.0 means select all points above the 90th percentile.
    copy : bool, default=False
        If True, returns a copy of the modified AnnData object.  
        Otherwise modifies the input object in place and returns None.

    Returns
    -------
    adata : anndata.AnnData or None
        If ``copy=True``, returns a copy of the AnnData with a new column  
        ``'LRC_<LRC_name>_<LRC_source>_unfiltered'`` in ``.obs``.
        Otherwise, modifies in place and returns None.

    Notes
    -----
    The resulting column is stored as a pandas ``Categorical`` with values {0, 1}.
    """
    
    # ==== Validate inputs ====
    assert LRC_name is not None, "Please provide an LRC_name."
    assert obs_name is not None, "Please provide an obs_name."

    # ==== Identify candidate cells ====
    threshold = np.percentile(adata.obs[obs_name].values, q=quantile)
    candidate_cells = adata.obs[obs_name].values.flatten() > threshold
    candidate_cells_int = candidate_cells.astype(int)
    candidate_cells_cat = pd.Categorical(candidate_cells_int)

    # ==== Store results ====
    key_name = f"LRC_{LRC_name}_{LRC_source}_unfiltered"
    adata.obs[key_name] = candidate_cells_cat

    print(f"Cells above the {quantile}% have been selected as candidates and stored in 'adata.obs['LRC_{LRC_name}_{LRC_source}_unfiltered']'.")

    return adata if copy else None

def LRC_cluster(
    adata: anndata.AnnData, 
    LRC_name: str = None,
    LRC_source: str = "marker",
    spatial_index: str = "spatial",
    density_cutoff: float = 10.0,
    delta_cutoff: float = 10.0,
    outlier_cutoff: float = 2.0, 
    fraction: float = 0.02,
    plot_savepath: str = None
):
    """
    Perform local density clustering on unfiltered LRC candidate points.

    This function applies a density–delta based clustering (as implemented in `pydpc.dpc.Cluster`)
    to identify candidate regions corresponding to a specific long-range channel (LRC).
    The results are visualized as density–delta plots and spatial cluster assignments.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (``n_obs × n_var``) containing spatial coordinates.
    LRC_name : str
        Name of the long-range channel (e.g. ``'Blood'`` or ``'CSF'``).
    LRC_source : str, default='marker'
        Type of source feature used for identifying LRC candidates (included in the key name).
    spatial_index : str, default='spatial'
        Key in ``adata.obsm`` storing spatial coordinates for clustering.
    density_cutoff : float, default=10.0
        Threshold for selecting cluster centers based on local density.
    delta_cutoff : float, default=10.0
        Threshold for selecting cluster centers based on delta distance.
    outlier_cutoff : float, default=2.0
        Density cutoff for filtering out low-density outliers.
    fraction : float, default=0.02
        Fraction of points relative to total used to estimate local density and delta.
    plot_savepath : str, optional
        Path to save the clustering diagnostic plots (e.g., ``'results/LRC_cluster.png'``).
        If None, the plot will be displayed interactively.

    Returns
    -------
    LRC_cluster : pydpc.dpc.Cluster
        The cluster object containing attributes such as `density`, `delta`,
        `membership`, and `outlier`, which can be used as input for
        :func:`mc.pp.LRC_filtered`.

    Notes
    -----
    The function requires that :func:`mc.pp.LRC_unfiltered` has been run beforehand,
    which stores unfiltered LRC candidates in ``adata.obs['LRC_<LRC_name>_<LRC_source>_unfiltered']``.
    """

    # ==== Validate inputs ====
    assert LRC_name is not None, "Please provide an LRC name."
    key = f"LRC_{LRC_name}_{LRC_source}_unfiltered"
    if key not in adata.obs.keys():
        raise KeyError("Please run the mc.pp.LRC_unfiltered function first.")

    # ==== Extract spatial coordinates ====
    LRC_cellsIndex = adata.obs[key].astype(bool)
    points = adata[LRC_cellsIndex,:].obsm[spatial_index].toarray().astype('double')

    # ==== Run local density clustering ====
    LRC_cluster = Cluster(points, fraction, autoplot=False)
    LRC_cluster.autoplot = False
    LRC_cluster.assign(density_cutoff, delta_cutoff)

    # ==== Identify outliers ====
    LRC_cluster.outlier = LRC_cluster.border_member
    LRC_cluster.outlier[LRC_cluster.density <= outlier_cutoff] = True
    LRC_cluster.outlier[LRC_cluster.density > outlier_cutoff] = False
    
    # ==== Plot results ====
    if points.shape[1] == 2:
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
    elif points.shape[1] == 3:
        fig, ax = plt.subplots(figsize=(5, 5))
        # Plot density vs. delta in the first subplot
        ax.scatter(LRC_cluster.density, LRC_cluster.delta, s=10)
        ax.plot([LRC_cluster.min_density, LRC_cluster.density.max()], [LRC_cluster.min_delta, LRC_cluster.min_delta], linewidth=2, color="red")
        ax.plot([LRC_cluster.min_density, LRC_cluster.min_density], [LRC_cluster.min_delta,  LRC_cluster.delta.max()], linewidth=2, color="red")
        ax.plot([outlier_cutoff, outlier_cutoff], [0,  LRC_cluster.delta.max()], linewidth=2, color="red", linestyle='--')
        ax.set_xlabel(r"density")
        ax.set_ylabel(r"delta / a.u.")
        ax.set_box_aspect(1)

    # ==== Save & Return ====
    if plot_savepath is not None:
        plt.savefig(plot_savepath)
        print(f"Plot saved to: {plot_savepath}")
    else:
        plt.show()

    # Return the cluster object
    return LRC_cluster

def LRC_filtered(
    adata: anndata.AnnData, 
    LRC_name: str = None,
    LRC_cluster = None,
    LRC_source: str = "marker",
    copy: bool = False
):
    """
    Assign final LRC (long-range channel) clusters after local density clustering.

    This function uses the cluster assignment results from :func:`mc.pp.LRC_cluster`
    to label candidate LRC points and remove outliers. The output is stored in
    ``adata.obs['LRC_<LRC_name>_<LRC_source>_filtered']``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix (``n_obs × n_var``).
    LRC_name : str
        Name of the long-range channel (e.g. ``'Blood'`` or ``'CSF'``).
    LRC_cluster : pydpc.dpc.Cluster
        The clustering object returned by :func:`mc.pp.LRC_cluster`.
    LRC_source : str, default='marker'
        Type of feature used for LRC identification (included in the key name).
    copy : bool, default=False
        If True, return a copy of the modified AnnData.
        Otherwise, modify in place and return None.

    Returns
    -------
    adata : anndata.AnnData or None
        The AnnData object with a new categorical column
        ``'LRC_<LRC_name>_<LRC_source>_filtered'`` in ``.obs``.
        Cluster numbers indicate LRC cluster IDs (starting from 1),
        while 0 indicates non-LRC or outlier points.
        Returns None if ``copy=False``.

    Notes
    -----
    This function should be run **after** both :func:`mc.pp.LRC_unfiltered` and :func:`mc.pp.LRC_cluster`. 
    """
    
    # ==== Validate inputs ====
    assert LRC_name is not None, "Please provide an LRC name."
    assert LRC_cluster is not None, "Please provide LRC_cluster."
    key = f"LRC_{LRC_name}_{LRC_source}_unfiltered"
    if key not in adata.obs.keys():
        raise KeyError(
            "Please run the 'mc.pp.LRC_unfiltered' and 'mc.pp.LRC_cluster' function first"
        )

    # ==== Compute filtered cluster ====
    newcluster = LRC_cluster.membership + 1
    newcluster[LRC_cluster.outlier] = 0

    # ==== Store results ====
    key_filtered = f"LRC_{LRC_name}_{LRC_source}_filtered"
    adata.obs[key_filtered] = adata.obs[key].astype(int)
    adata.obs[key_filtered][adata.obs[key_filtered] == 1] = newcluster
    adata.obs[key_filtered] = adata.obs[key_filtered].astype('category')

    print(
        f"Candidate points for {LRC_name} LRC are clustered and outliers are removed. "
        f"LRC points are stored in 'adata.obs['LRC_{LRC_name}_{LRC_source}_filtered']'."
    )

    return adata if copy else None