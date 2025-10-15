import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import itertools
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import networkx as nx
import matplotlib as mpl
from pydpc import Cluster
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial import distance_matrix
from multiprocessing import Pool, cpu_count
import plotly.express as px
import scipy.sparse as sp

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

    return adata.copy() if copy else None

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

    return adata.copy() if copy else None

def load_barrier_segments(
    csv_path: str = None,
    coord_cols = ("axis-2", "axis-1"),
    close_polygons: bool = True,
    scale: float = None
):
    """
    Parse Napari shapes CSV and extract barrier line segments.

    This function converts a Napari shapes `.csv` file (usually exported from Napari's
    "Shapes" layer) into a list of 2D line segments represented as coordinate pairs.
    Each shape is grouped by its `index` and its vertices ordered by `vertex-index`.

    Parameters
    ----------
    csv_path : str
        Path to the Napari shapes CSV file.
    coord_cols : tuple of str, default=('axis-2', 'axis-1')
        Column names representing the coordinate axes in the CSV.
        The order is typically ('axis-2', 'axis-1') = (Y, X).
    close_polygons : bool, default=True
        Whether to close polygonal shapes by connecting the last vertex to the first.
    scale : float, optional
        Scaling factor applied to all coordinates.  
        For example, set `scale=0.5` to convert from pixel to micrometer units.

    Returns
    -------
    segs : list of tuple
        A list of line segments, each represented as
        `[((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)), ...]`.

    Notes
    -----
    The input CSV should contain at least the following columns:
    `['index', 'vertex-index', 'shape-type', 'axis-2', 'axis-1']`.
    """
    
    # ==== Read and group CSV ====
    df = pd.read_csv(csv_path)
    segs = []

    # ==== Extract line segments ====
    for idx, g in df.groupby("index", sort=True):
        g = g.sort_values("vertex-index")
        shape = g["shape-type"].iloc[0].lower()
        P = g[list(coord_cols)].to_numpy(dtype=float)
        if len(P) < 2: 
            continue
        for a, b in zip(P[:-1], P[1:]):
            segs.append((tuple(a), tuple(b)))
        if shape == "polygon" and close_polygons:
            segs.append((tuple(P[0]), tuple(P[1])))
    
    # ==== Apply scaling ====
    if scale is not None:
        segs = [((a[0]*scale, a[1]*scale), (b[0]*scale, b[1]*scale)) for a, b in segs]

    return segs

def _orient(a, b, c):
    """Compute signed area orientation (cross product) of triangle (a, b, c)."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_segment(a, b, c, tol=1e-9):
    """Check if point c lies on segment ab (within tolerance)."""
    return (min(a[0], b[0]) - tol <= c[0] <= max(a[0], b[0]) + tol and
            min(a[1], b[1]) - tol <= c[1] <= max(a[1], b[1]) + tol and
            abs(_orient(a, b, c)) <= tol)

def _segments_intersect(p1, p2, q1, q2, tol=1e-9, block_touch=True):
    """Return True if line segments (p1,p2) and (q1,q2) intersect."""
    o1 = _orient(p1, p2, q1); o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1); o4 = _orient(q1, q2, p2)
    # proper intersection
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    # collinear/touching
    if block_touch:
        if abs(o1) <= tol and _on_segment(p1, p2, q1, tol): return True
        if abs(o2) <= tol and _on_segment(p1, p2, q2, tol): return True
        if abs(o3) <= tol and _on_segment(q1, q2, p1, tol): return True
        if abs(o4) <= tol and _on_segment(q1, q2, p2, tol): return True
    return False

def compute_edge_if_visible(p1, p2, line_segments, tol=1e-9, block_touch=True):
    """Check visibility between two points, return (p1, p2, distance) or (p1, p2, None) if blocked."""
    for q1, q2 in line_segments:
        # quick AABB reject to speed up
        if (max(p1[0], p2[0]) < min(q1[0], q2[0]) or
            max(q1[0], q2[0]) < min(p1[0], p2[0]) or
            max(p1[1], p2[1]) < min(q1[1], q2[1]) or
            max(q1[1], q2[1]) < min(p1[1], p2[1])):
            continue
        if _segments_intersect(p1, p2, q1, q2, tol=tol, block_touch=block_touch):
            return (p1, p2, None)
    w = float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))
    return (p1, p2, w)

def _init_visible_worker(line_segments):
    """Store barrier line segments for multiprocessing workers."""
    global _LINE_SEGS
    _LINE_SEGS = line_segments

def _edge_task(task):
    """Visibility check task for parallel computation."""
    p1_org, p2_org = task
    vis = compute_edge_if_visible(p1_org, p2_org, _LINE_SEGS, tol=1e-9, block_touch=True)
    if vis is None or vis[2] is None:
        return (p1_org, p2_org, None)
    w = float(np.linalg.norm(np.asarray(p1_org) - np.asarray(p2_org)))
    return (p1_org, p2_org, w)

def build_visible_graph(
    adata: anndata.AnnData,
    barrier_segments: list = None,
    use_parallel: bool = True,
    n_jobs: int = -1,
    copy: bool = False
):
    """
    Build a visibility graph from spatial coordinates with barrier constraints.
    Any edge intersecting a barrier segment is excluded.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing ``adata.obsm["spatial"]`` of shape (n, 2)
        in the same coordinate system as barriers.
    barrier_segments : list
        List of barrier segments [((x1, y1), (x2, y2)), ...], typically obtained
        from :func:`load_barrier_segments`.
    use_parallel : bool, default=True
        Whether to use multiprocessing for visibility computation.
    n_jobs : int, default=-1
        Number of CPU cores to use; -1 means use all available.
    copy : bool, default=False
        If True, return a copy of AnnData; otherwise modify in place.

    Returns
    -------
    AnnData or None
        Stores CSR adjacency in `adata.obsp['graph_visible']` and DataFrame of blocked pairs in `adata.uns['non_visible_pair']`.
    """
    # ---- Basic checks ----
    if "spatial" not in adata.obsm:
        raise ValueError("`adata.obsm['spatial']` is missing.")
    if barrier_segments is None:
        raise ValueError("`barrier_segments` must be provided as a list of ((x,y),(x,y)) tuples.")
    if adata.obsm["spatial"].shape[1] != 2:
        raise ValueError("`adata.obsm['spatial']` must have shape (n, 2).")

    # Work on a view or a copy?
    ad = adata.copy() if copy else adata
    coords = np.asarray(ad.obsm["spatial"], dtype=float)
    spots_positions = [tuple(xy) for xy in coords]
    pos_to_idx = {tuple(xy): i for i, xy in enumerate(coords)}

    # ---- Build pairwise task list in continuous coords ----
    task_list = list(itertools.combinations(spots_positions, 2))
    rows, cols, data = [], [], []
    non_visible_pair_idx = []

    # ---- Configure workers ----
    n_workers = cpu_count() if (n_jobs == -1 or n_jobs is None) else max(1, int(n_jobs))

    # ---- Run computation ----
    if use_parallel:
        chunksize = max(1, len(task_list) // (n_workers * 8) or 1)
        with Pool(processes=n_workers,
                  initializer=_init_visible_worker,
                  initargs=(barrier_segments,)) as pool, \
             tqdm(total=len(task_list), desc=f"  Building visible graph with paralleling {n_workers} CPU cores...", dynamic_ncols=True) as pbar:
            try:
                for u_org, v_org, w in pool.imap_unordered(_edge_task, task_list, chunksize=chunksize):
                    i, j = pos_to_idx[u_org], pos_to_idx[v_org]
                    if w is not None:
                        rows.extend([i, j]); cols.extend([j, i]); data.extend([w, w])
                    else:
                        non_visible_pair_idx.append((i, j, u_org, v_org))
                    pbar.update(1)
            except KeyboardInterrupt:
                pool.terminate()
                raise
            else:
                pool.close()
            pool.join()
    else:
        # Single-process path
        global _LINE_SEGS
        _LINE_SEGS = barrier_segments
        for task in tqdm(task_list, desc="  Building visible graph...", dynamic_ncols=True):
            u_org, v_org, w = _edge_task(task)
            i, j = pos_to_idx[u_org], pos_to_idx[v_org]
            if w is not None:
                rows.extend([i, j]); cols.extend([j, i]); data.extend([w, w])
            else:
                non_visible_pair_idx.append((i, j, u_org, v_org))

    # ---- Save into CSR adjacency (symmetric) ----
    n = ad.n_obs
    if len(rows) == 0:
        adj = sp.csr_matrix((n, n), dtype=np.float32)
    else:
        adj = sp.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(n, n))

    non_visible_pair_idx_df = pd.DataFrame(
        [(i, j, u[0], u[1], v[0], v[1]) for i, j, u, v in non_visible_pair_idx],
        columns=['coords_index_u', 'coords_index_v', 'coords_u_x', 'coords_u_y', 'coords_v_x', 'coords_v_y']
    )

    ad.obsp['graph_visible'] = adj
    ad.uns['non_visible_pair'] = non_visible_pair_idx_df

    return ad if copy else None

def load_graph_visible_from_obsp(adata, key=None, coords="spatial"):
    """
    Rebuild a NetworkX Graph from `adata.obsp[key]`.
    - Edge weights are taken from the sparse matrix values.
    - `label='spatial'` relabels nodes to tuple(coords) from `adata.obsm['spatial']`;
      use `label='index'` to keep integer node ids (0..n-1).
    """
    A = adata.obsp[key]
    if not sp.issparse(A):
        # Convert dense to CSR for speed, if needed
        A = sp.csr_matrix(A)
    # Build undirected weighted graph from sparse matrix
    G = nx.from_scipy_sparse_array(A, edge_attribute="weight")
    mapping = {i: tuple(xy) for i, xy in enumerate(np.asarray(adata.obsm[coords]))}
    G = nx.relabel_nodes(G, mapping)

    return G

def nx_graph_to_csr(G, n_nodes):
    # Build symmetric COO from undirected edges; keep weights as float32
    rows, cols, data = [], [], []
    for u, v, d in G.edges(data=True):
        if u == v:
            continue  # drop self-loops
        w = float(d.get("weight", 1.0))
        rows.extend([u, v])   # undirected -> insert both directions
        cols.extend([v, u])
        data.extend([w, w])
    if not rows:
        return sp.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    A = sp.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(n_nodes, n_nodes))

    return A

def build_visible_graph_knn(
    adata: anndata.AnnData,
    k_neighb: int = 5,
    copy: bool = False
):
    """
    Construct a k-nearest neighbor (k-NN) visibility graph from an existing visibility graph.

    This function trims the precomputed visibility graph in ``adata.obsp["graph_visible"]`` 
    to retain only the k-nearest neighbors (based on edge weights) for each node, 
    resulting in a sparse but structured graph.

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data object containing:
        - ``adata.obsm["spatial"]`` : Spatial coordinates (n × 2)
        - ``adata.obsp["graph_visible"]`` : CSR matrix of visible edge weights, 
          typically produced by :func:`mc.pp.build_visible_graph`.
    k_neighb : int, default=5
        Number of nearest neighbors to retain for each node.
    copy : bool, default=False
        Whether to return a modified copy of AnnData instead of modifying in place.

    Returns
    -------
    anndata.AnnData or None
        If ``copy=True``, returns a new AnnData object.  
        Adds the following:
        - ``adata.obsp[f"graph_visible_kneighb_{k_neighb}"]`` : CSR matrix of k-NN edges.
    """
    # ---- Check inputs ----
    if not "graph_visible" in adata.obsp.keys():
        raise KeyError("Missing 'graph_visible' in adata.obsp, please run the 'mc.pp.build_visible_graph' function first")
    
    # ---- Prepare workspace ----
    G_visible = load_graph_visible_from_obsp(adata, key="graph_visible")
    G = G_visible.copy()

    edges_to_remove = []

    for node in G.nodes:
        # Retrieve all neighbors and their edge weights
        neighbors = [(neighbor, G[node][neighbor]['weight']) for neighbor in G.neighbors(node)]
        # Sort by edge weight (distance), keeping only the k-nearest neighbors
        neighbors.sort(key=lambda x: x[1])
        keep_edges = {neighbor[0] for neighbor in neighbors[:k_neighb]}

        # Identify edges to remove
        for neighbor in G.neighbors(node):
            if neighbor not in keep_edges:
                edges_to_remove.append((node, neighbor))

    # Remove unnecessary edges
    G.remove_edges_from(edges_to_remove)

    coords = np.asarray(adata.obsm["spatial"])
    spots_positions = [tuple(xy) for xy in coords]
    pos_to_idx = {pos: i for i, pos in enumerate(spots_positions)}
    G = nx.relabel_nodes(G, pos_to_idx)
    A_knn = nx_graph_to_csr(G, adata.n_obs)

    adata.obsp[f'graph_visible_kneighb_{k_neighb}'] = A_knn

    return adata.copy() if copy else None

def init_graph(G_kneigh):
    """
    Initializes the global variable G_kneigh for multiprocessing.

    This function sets a global variable G_kneigh_global, allowing all worker 
    processes in a multiprocessing pool to access the same graph without 
    repeatedly copying it.
    """
    global G_kneigh_global
    G_kneigh_global = G_kneigh

def compute_shortest_path(args):
    """
    Computes the shortest path distance between two non-visible points.

    This function calculates the shortest path between two spatial points using 
    Dijkstra's algorithm on the global k-nearest neighbor graph (G_kneigh_global). 
    If no path exists, the distance is set to infinity.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - i (int): Index of the first point in the distance matrix.
        - j (int): Index of the second point in the distance matrix.
        - pos1 (tuple): Coordinates of the first point.
        - pos2 (tuple): Coordinates of the second point.

    Returns
    -------
    tuple
        A tuple (i, j, shortest_path_length) where:
        - i (int): Index of the first point.
        - j (int): Index of the second point.
        - shortest_path_length (float): The shortest path distance between the points.
    """
    i, j, pos1, pos2 = args
    global G_kneigh_global
    try:
        shortest_path_length = nx.dijkstra_path_length(G_kneigh_global, source=pos1, target=pos2, weight="weight")
        return i, j, pos1, pos2, shortest_path_length
    except nx.NetworkXNoPath:
        return i, j, pos1, pos2, np.inf  # No path exists

def compute_shortest_path_single_source(args):
    """
    Computes single-source shortest path lengths from a given node in a global graph.

    This function is designed for use in parallel processing to compute shortest path 
    lengths from a source node to all reachable nodes using Dijkstra's algorithm. 
    It depends on a globally shared NetworkX graph (`G_kneigh_global`), which should 
    be initialized using `init_graph()` prior to calling this function in a multiprocessing pool.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - m : int
            The index of the source node (typically local index in a cluster or spatial list).
        - pos1 : tuple
            The spatial coordinate (node position) of the source.

    Returns
    -------
    tuple
        A tuple of (m, pos1, lengths), where:
        - m : int
            The same index passed in for reference.
        - pos1 : tuple
            The same source coordinate.
        - lengths : dict
            A dictionary mapping each reachable node position to its shortest path 
            distance from the source. If no path exists, an empty dictionary is returned.
    """
    m, pos1 = args
    global G_kneigh_global
    try:
        lengths = nx.single_source_dijkstra_path_length(G_kneigh_global, source=pos1, weight="weight")
    except nx.NetworkXNoPath:
        lengths = {}
    return m, pos1, lengths

def build_distance_matrix(adata: anndata.AnnData,
                          k_neighb: int = 5,
                          use_parallel: bool = True, 
                          n_jobs: int = -1,
                          copy: bool = False):
    """
    Constructs an updated distance matrix incorporating shortest paths for non-visible points.

    This function initializes a distance matrix based on Euclidean distances and updates 
    distances for non-visible points using shortest paths computed from the k-nearest 
    neighbor graph (G_kneigh).

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing spatial information.
        - `adata.obsm["spatial"]` : ndarray of shape (n_spots, 2)
            The spatial coordinates of the spots.
    non_visible_pair : list of tuple
        A list of spatial coordinate pairs (pos1, pos2) representing 
        non-visible connections.
    G_kneigh : networkx.Graph
        The k-nearest neighbor graph ensuring connectivity.
    use_parallel : bool, optional
        Whether to use parallel processing for shortest path computation (default is True).
    n_jobs : int, optional
        Number of parallel jobs to use for computation (default is -1, using all available CPUs).

    Returns
    -------
    np.ndarray
        A symmetric distance matrix where unreachable distances are replaced 
        with shortest paths from G_kneigh.
    """
    
    if not "non_visible_pair" in adata.uns.keys():
        raise KeyError("Missing 'non_visible_pair' in adata.uns, please run the 'mc.pp.build_visible_graph' function first")  
    if not f'graph_visible_kneighb_{k_neighb}' in adata.obsp.keys():
        raise KeyError(f"Missing 'graph_visible_kneighb_{k_neighb}' in adata.obsp, please run the 'mc.pp.build_visible_graph_knn' function first")

    non_visible_pair = adata.uns['non_visible_pair'].copy()
    G_kneigh = load_graph_visible_from_obsp(adata, key=f'graph_visible_kneighb_{k_neighb}')

    # Create a mapping from spatial coordinates to matrix indices
    spots_positions = [tuple(coord) for coord in adata.obsm["spatial"]]
    position_to_index = {pos: i for i, pos in enumerate(spots_positions)}

    # Compute the initial Euclidean distance matrix
    dis_mat = distance_matrix(spots_positions, spots_positions)
    updated_dis_mat = dis_mat.copy()

    # Prepare tasks for shortest path computation
    task_list = list(zip(
        non_visible_pair['coords_index_u'],
        non_visible_pair['coords_index_v'],
        zip(non_visible_pair['coords_u_x'], non_visible_pair['coords_u_y']),
        zip(non_visible_pair['coords_v_x'], non_visible_pair['coords_v_y'])
    ))

    results = []
    if use_parallel:
        with Pool(processes=n_jobs, initializer=init_graph, initargs=(G_kneigh,)) as pool:
            with tqdm(total=len(task_list), desc="  Computing shortest paths for non-visible points", dynamic_ncols=True) as pbar:
                for result in pool.imap_unordered(compute_shortest_path, task_list):
                    results.append(result)
                    pbar.update(1)  # Update progress bar for each completed task
    else:
        for args in tqdm(task_list, desc="  Computing shortest paths for non-visible points", dynamic_ncols=True):
            results.append(compute_shortest_path(args))

    # Update the distance matrix with shortest path values
    for i, j, pos1, pos2, shortest_path_length in results:
        updated_dis_mat[i, j] = shortest_path_length
        updated_dis_mat[j, i] = shortest_path_length  # Ensure symmetry
    
    adata.obsp['spatial_distance_LRC_base'] = updated_dis_mat

    return adata if copy else None


def compute_costDistance(
    adata: anndata.AnnData,
    LRC_type: list = None,
    LRC_strength: list = None,
    LRC_source: str = "marker",
    dis_thr: float = 50.0,
    k_neighb: int = 5,
    barrier: bool = False,
    barrier_segments: list = None,
    spatial_3d: bool = False,                        
    use_parallel: bool = True,
    n_jobs: int = -1,
    copy: bool = False
):
    """
    Compute LRC-embedding cost distance based on visibility and local connectivity,
    supporting both 2D and 3D spatial coordinates.

    Parameters
    ----------
    adata
        AnnData object containing spatial coordinates and LRC annotations.
    LRC_type
        List of long-range communication (LRC) types to process, e.g. ["CSF", "Blood"].
    LRC_strength
        List of reweighting strengths corresponding to each LRC type (same length as LRC_type).
    LRC_source
        Labeling source in `adata.obs`, either "marker" (default) or "manual".
    dis_thr
        Distance threshold for defining neighborhood around each LRC (default: 50.0).
    k_neighb
        Number of nearest neighbors for within-cluster graph construction (default: 5).
    barrier
        Whether to include visibility barriers (default: False).
    barrier_segments
        List of ((x1, y1), (x2, y2)) tuples describing physical barriers.
    spatial_3d
        Use 3D spatial coordinates (`adata.obsm["spatial_3d"]`) instead of 2D (default: False).
    use_parallel
        Whether to parallelize shortest-path computation (default: True).
    n_jobs
        Number of parallel jobs; -1 uses all CPUs (default: -1).
    copy
        Return a copy instead of modifying in place (default: False).

    Returns
    -------
    anndata.AnnData or None
        If copy=True, returns a new AnnData with updated distances.  
        Stores:
        - ``adata.obsp['spatial_distance_LRC_base']``
        - ``adata.obsp['spatial_distance_LRC_<type>']``
    """
    
    ### Check if the input is a valid.
    if (LRC_type is None) and (LRC_strength is None):
        print("No 'LRC_type' and 'LRC_strength' are provided. Long-range communication will not be included in subsequent analysis.")

    if (LRC_type is not None) and ((LRC_strength is None) or (len(LRC_strength) != len(LRC_type))):
        raise ValueError("Please provide a list of LRC_strength values that matches the length of LRC_type.")
    
    if barrier and 'barrier' not in adata.obs:
        raise KeyError("adata.obs['barrier'] missing while barrier=True.")
    
    ### Check parallel settings
    n_jobs = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
    
    ### Check if the dimension of spatial coordinates is 2D or 3D
    spatial_key = 'spatial_3d' if spatial_3d else 'spatial'
    spatial_coords = adata.obsm[spatial_key]

    # ============ Baseline spatial distance without LRC ============
    if 'spatial_distance_LRC_base' not in adata.obsp:
        print("Computing baseline spatial distance without LRC.")        
        if barrier:
            print("  Barrier condition is considered")
            build_visible_graph(adata=adata, barrier_segments=barrier_segments, use_parallel=use_parallel, n_jobs=n_jobs)
            build_visible_graph_knn(adata=adata, k_neighb=k_neighb)
            build_distance_matrix(adata=adata, k_neighb=k_neighb, use_parallel=use_parallel, n_jobs=n_jobs)
        else:
            print("  Barrier condition is not considered")
            adata.obsp['spatial_distance_LRC_base'] = np.array(distance_matrix(spatial_coords, spatial_coords), dtype=np.float32)         
    dis_mat = adata.obsp['spatial_distance_LRC_base'].copy()
    
    if LRC_type is None:
        print("No LRC_type provided — skipping LRC embedding computation, 'adata.obsp['spatial_distance_LRC_base']' has been saved.")
        return adata if copy else None
    
    # ============ Spatial distance with LRCs ============
    print("Computing spatial distance with LRCs.")
    for idx_LRC, LRC_element in enumerate(LRC_type):
        print(f"  Processing LRC type: {LRC_element}")
        strength = LRC_strength[idx_LRC]

        if LRC_source == "marker":
            key_LRC = f'LRC_{LRC_element}_marker_filtered'
            if key_LRC not in adata.obs:
                raise KeyError(f"Missing adata.obs['{key_LRC}'] — please run LRC_unfiltered → cluster → filtered.")
        elif LRC_source == "manual":
            key_LRC = f'LRC_{LRC_element}_manual_filtered'
            if key_LRC not in adata.obs:
                raise KeyError(f"Missing adata.obs['{key_LRC}'] — please provide manually annotated LRC.")
        else:
            raise ValueError("LRC_source_mode must be one of ['marker', 'manual']")
        
        ## ============ Recoding close points next to each LRC ============
        LRC_subcluster = sorted(set(adata.obs[key_LRC]) - {0})        
        record_closepoint = np.zeros((len(adata.obs), len(LRC_subcluster)))
        spot_close_LRC, spot_close_LRC_type = [], []

        for ispot in range(dis_mat.shape[0]):
            spot_close_ind = dis_mat[ispot] < dis_thr
            temp = adata.obs[key_LRC][spot_close_ind]
            if np.any(temp != 0):
                spot_close_LRC.append(ispot)
                ispot_LRC_type = sorted(set(temp[temp != 0]))
                for t in ispot_LRC_type:
                    record_closepoint[ispot, int(t) - 1] = 1
                spot_close_LRC_type.append(ispot_LRC_type)
        spot_close_LRC = np.array(spot_close_LRC)

        for icluster in LRC_subcluster:
            adata.obs[f'LRC_{LRC_element}_closepoint_cluster{icluster}'] = record_closepoint[:, int(icluster) - 1]
        
        ## ============ Computing shortest path between two points in LRCs ============
        if all(f'LRC_shortest_{LRC_element}_dist{dis_thr}_cluster{icluster}' in adata.obsp for icluster in LRC_subcluster):
            print(f"    Using cached matrix paths in adata.obsp for LRC type {LRC_element} under dis_thr {dis_thr}.")
        else:
            ### ============ Construct graph for each LRC subcluster ============ ###
            print(f"    Constructing graph for each LRC subcluster.")
            G_LRC = {}
            for icluster in LRC_subcluster:
                icluster = int(icluster)
                G = nx.Graph()
                LRC_coords_icluster = spatial_coords[adata.obs[key_LRC] == icluster]
                for pos in map(tuple, LRC_coords_icluster):
                    G.add_node(pos)
                dis_mat_local = dis_mat[adata.obs[key_LRC] == icluster,adata.obs[key_LRC] == icluster]
                for i, pos in enumerate(map(tuple, LRC_coords_icluster)):
                    for j in np.argsort(dis_mat_local[i])[1:k_neighb+1]:
                        G.add_edge(pos, tuple(LRC_coords_icluster[j]), weight=dis_mat_local[i][j])
                if not nx.is_connected(G):
                    comps = list(nx.connected_components(G))
                    for a, b in itertools.combinations(comps, 2):
                        A = np.array(list(a))
                        B = np.array(list(b))
                        dis = distance_matrix(A, B)
                        i, j = np.unravel_index(dis.argmin(), dis.shape)
                        G.add_edge(tuple(A[i]), tuple(B[j]), weight=dis[i, j])
                G_LRC[f'{icluster}'] = G

            ### ============ Compute shortest path between two points in LRCs ============ ###
            print(f"    Computing shortest path between two points in {LRC_element}.")
            for i, icluster in enumerate(LRC_subcluster):
                icluster = int(icluster)
                print(f"      For the cluster {icluster} in {LRC_element}.")
                G_LRC_subcluster = G_LRC[f'{icluster}']
                LRC_index_icluster = np.where(adata.obs[key_LRC] == icluster)[0]
                LRC_coords_icluster = spatial_coords[LRC_index_icluster]

                task_list = [
                    (m, tuple(LRC_coords_icluster[m])) for m in range(len(LRC_coords_icluster))
                ]

                dis_LRC_shortest = np.zeros((adata.n_obs, adata.n_obs))
                if use_parallel:
                    with Pool(n_jobs, initializer=init_graph, initargs=(G_LRC_subcluster,)) as pool:
                        with tqdm(total=len(task_list)) as pbar:
                            for m, p1, lengths in pool.imap_unordered(compute_shortest_path_single_source, task_list):
                                i_global = LRC_index_icluster[m]
                                for n in range(m+1, len(LRC_coords_icluster)):
                                    p2 = tuple(LRC_coords_icluster[n])
                                    j_global = LRC_index_icluster[n]
                                    d = lengths.get(p2, np.inf)
                                    dis_LRC_shortest[i_global, j_global] = d
                                    dis_LRC_shortest[j_global, i_global] = d
                                pbar.update(1)
                else:
                    init_graph(G_LRC_subcluster)
                    for m, p1 in tqdm(task_list):
                        _, _, lengths = compute_shortest_path_single_source((m, p1))
                        i_global = LRC_index_icluster[m]
                        for n in range(m+1, len(LRC_coords_icluster)):
                            p2 = tuple(LRC_coords_icluster[n])
                            j_global = LRC_index_icluster[n]
                            d = lengths.get(p2, np.inf)
                            dis_LRC_shortest[i_global, j_global] = d
                            dis_LRC_shortest[j_global, i_global] = d
                adata.obsp[f'LRC_shortest_{LRC_element}_dist{dis_thr}_cluster{icluster}'] = sp.csr_matrix(dis_LRC_shortest.astype(np.float32))

        ## ============ Rearranging distance matrix ============
        print(f"    Incorporating LRC strength of '{LRC_element}' into cost distance matrix (strength = {strength}).")
        dis_mat_LRC = np.zeros((len(LRC_subcluster)+1, *dis_mat.shape))
        dis_mat_LRC[0] = dis_mat

        for idx, icluster in enumerate(LRC_subcluster):
            dis_LRC_shortest = adata.obsp[f'LRC_shortest_{LRC_element}_dist{dis_thr}_cluster{icluster}']
            if sp.issparse(dis_LRC_shortest):
                dis_LRC_shortest = dis_LRC_shortest.toarray()
            idx_mask = np.array([i for i, types in zip(spot_close_LRC, spot_close_LRC_type) if icluster in types])
            dis2LRC = []
            closest_spot_idx = []
            LRC_filter = adata.obs[key_LRC] == icluster
            for ispot in idx_mask:
                spot_close_ind = dis_mat[ispot] < dis_thr
                spot_ids = np.where(spot_close_ind & LRC_filter)[0]
                if not len(spot_ids): continue
                min_id = spot_ids[np.argmin(dis_mat[ispot, spot_ids])]
                closest_spot_idx.append(min_id)
                dis2LRC.append(dis_mat[ispot, min_id])
            dis2LRC = np.array(dis2LRC)
            dis_LRC = np.add.outer(dis2LRC, dis2LRC)
            dis_mat_LRC_path = dis_LRC_shortest[np.ix_(closest_spot_idx, closest_spot_idx)]
            reweighted = dis_mat.copy()
            reweighted[np.ix_(idx_mask, idx_mask)] = dis_LRC + dis_mat_LRC_path / strength
            dis_mat_LRC[idx+1] = reweighted

        print(f"adata.obsp['spatial_distance_LRC_{LRC_element}'] has been saved.")
        adata.obsp[f'spatial_distance_LRC_{LRC_element}'] = np.array(np.min(dis_mat_LRC, axis=0), dtype=np.float32)

    print("Finished!")
    return adata.copy() if copy else None

def global_intensity_scaling(
    adata_ref: anndata.AnnData,
    adata_target: anndata.AnnData,
    method: str = 'tic',
    scales: float = 1e-5
):
    """
    Perform global intensity scaling of `adata_target` to match `adata_ref`,
    using either total ion current (TIC) or root-mean-square (RMS) normalization.

    Parameters
    ----------
    adata_ref
        Reference dataset for scaling (e.g., negative ion mode).
    adata_target
        Target dataset to be scaled (e.g., positive ion mode).
    method
        Scaling method to use:
        - `'tic'`: scale by total ion current (sum of all intensities)
        - `'rms'`: scale by root-mean-square of intensities
    scales
        Optional global scaling factor applied to both datasets (default: 1e-5).

    Returns
    -------
    adata_ref : anndata.AnnData
        Scaled reference dataset.
    adata_target : anndata.AnnData
        Scaled target dataset.
    """
    # Extract dense arrays for computation
    if hasattr(adata_ref.X, "toarray"):
        ref_data = adata_ref.X.toarray()
    else:
        ref_data = adata_ref.X.copy()
    if hasattr(adata_target.X, "toarray"):
        tgt_data = adata_target.X.toarray()
    else:
        tgt_data = adata_target.X.copy()
    
    if method == 'tic':
        # Compute global TIC for reference and target
        global_ref = np.sum(ref_data)
        global_tgt = np.sum(tgt_data)
    elif method == 'rms':
        # Compute global RMS for reference and target
        global_ref = np.sqrt(np.mean(np.square(ref_data)))
        global_tgt = np.sqrt(np.mean(np.square(tgt_data)))
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'tic' or 'rms'.")
    
    # Compute scale factor, avoid division by zero
    scale_factor = float(global_ref) / float(global_tgt) if global_tgt != 0 else 1.0
    
    # Apply constant scaling to the entire target matrix
    if hasattr(adata_target.X, "multiply"):
        adata_target.X = adata_target.X.multiply(scale_factor * scales)
    else:
        adata_target.X = adata_target.X * scale_factor * scales
    
    if hasattr(adata_ref.X, "multiply"):
        adata_ref.X = adata_ref.X.multiply(scales)
    else:
        adata_ref.X = adata_ref.X * scales

    return adata_ref, adata_target