# ============================================================
import numpy as np
import pandas as pd
import anndata
# ============================================================

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