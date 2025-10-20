import plotly
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def get_cmap_qualitative(cmap_name):
    if cmap_name == "Plotly":
        cmap = plotly.colors.qualitative.Plotly
    elif cmap_name == "Alphabet":
        cmap = plotly.colors.qualitative.Alphabet
    elif cmap_name == "Light24":
        cmap = plotly.colors.qualitative.Light24
    elif cmap_name == "Dark24":
        cmap = plotly.colors.qualitative.Dark24
    return cmap

def auto_quiver_scale(Xpos, Vec, frac=0.05):
    """
    Estimate a good quiver scale so arrows are visible without hand-tuning.
    - Xpos : (N,2) positions of arrows
    - Vec  : (N,2) vectors
    - frac : target arrow length = frac * max(x/y span)
    """
    # remove NaN
    m = np.isfinite(Vec).all(axis=1)
    if not np.any(m):
        return 1.0

    L = np.linalg.norm(Vec[m], axis=1)
    Lref = np.nanpercentile(L, 90)   # use 90th percentile length as reference

    span = max(Xpos[:,0].ptp(), Xpos[:,1].ptp())  # plot span
    target = span * frac

    return Lref / target if Lref > 0 else 1.0

def plot_cell_signaling(
    # ==== 1. Core data ====
    coords,
    Vector,
    signal_sum,
    summary = 'receiver', # {'sender','receiver'}
    plot_method = "cell", # {'cell','grid'}

    # ==== 2. Background & grouping ====
    adata = None,
    background = 'image', # {'image', 'group', 'summary'}
    library_id = None,
    group_name = None,
    group_cmap = None,
    background_legend = False,
    cmap = "coolwarm",
    normalize_summary_quantile = 0.995,
    ndsize = 1,
    vmin = None,
    vmax = None,    

    # ==== 3. Grid / smoothing parameters ====
    grid_density = 1,
    grid_knn = None,
    grid_scale = 1.0,
    grid_thresh = 1.0,

    # ==== 4. Arrow drawing & scaling ====
    arrow_color = "tab:blue",
    arrow_width = 0.005,
    largest_arrow = 0.05,

    # ==== 5. Save / axes ====
    title = None,
    plot_savepath = None,
    ax = None
):
    """Internal helper function used by plot_communication_flow()."""
    ndcolor = signal_sum.copy()
    ndcolor_percentile = np.percentile(ndcolor, normalize_summary_quantile*100)
    ndcolor[ndcolor > ndcolor_percentile] = ndcolor_percentile
    
    V_cell = Vector.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan

    if summary == "sender":
        X_vec = coords
        quiver_pivot = "tail"
    elif summary == "receiver":
        X_vec = coords
        quiver_pivot = "tip"

    if plot_method == "grid":    
        # --- rectangular grid with consistent step (linspace includes endpoints) ---
        xl, xr = np.min(coords[:, 0]), np.max(coords[:, 0])
        yl, yr = np.min(coords[:, 1]), np.max(coords[:, 1])
        xl -= 0.02 * (xr - xl); xr += 0.02 * (xr - xl)
        yl -= 0.02 * (yr - yl); yr += 0.02 * (yr - yl)

        ngrid_x = max(2, int(50 * grid_density))
        # step along x with ngrid_x points (inclusive endpoints)
        gridsize = (xr - xl) / max(ngrid_x - 1, 1)
        # match y with the same step; +1 because endpoints are included
        ngrid_y = max(2, int(round((yr - yl) / gridsize)) + 1)

        x_grid = np.linspace(xl, xr, ngrid_x)
        y_grid = np.linspace(yl, yr, ngrid_y)
        XX, YY = np.meshgrid(x_grid, y_grid)
        grid_pts = np.c_[XX.ravel(), YY.ravel()]
        G = grid_pts.shape[0]

        # --- kNN on cells for each grid point (Euclidean) ---
        if grid_knn is None:
            grid_knn = int(max(1, coords.shape[0] // 50))
        grid_knn = int(np.clip(grid_knn, 1, coords.shape[0]))

        nn_mdl = NearestNeighbors(algorithm="kd_tree", metric="euclidean")
        nn_mdl.fit(coords)
        dis_eu, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn, return_distance=True)  # (G,k), (G,k)

        # --- anchor cell for each grid point (to use cell–cell geodesic distances) ---
        dis_anchor, anchor_idx_all = nn_mdl.kneighbors(grid_pts, n_neighbors=1, return_distance=True)
        dis_anchor = dis_anchor.ravel()
        anchor_idx_all = anchor_idx_all.ravel().astype(int)  # (G,)

        # --- (optional) geodesic/visible filtering using cell×cell distance matrix ---
        # If available (e.g. adata.obsp['spatial_distance_LRC_base']), use it to filter neighbors.
        dist_mat = None
        if 'spatial_distance_LRC_base' in getattr(adata, "obsp", {}):
            dm = adata.obsp['spatial_distance_LRC_base']
            dist_mat = dm.toarray() if hasattr(dm, "toarray") else np.asarray(dm)

        # geodesic radius comparable to the grid step (you can parameterize it)
        geodesic_radius = max(gridsize * grid_scale, 1e-9)

        # --- Gaussian weights on Euclidean distances (for smoothing) ---
        scale_gauss = max(gridsize * grid_scale, 1e-9)
        w = norm.pdf(x=dis_eu, scale=scale_gauss)  # (G,k)

        if dist_mat is not None:
            # keep neighbors that are geodesically reachable and within radius from the anchor
            dist_anchor_nb = dist_mat[anchor_idx_all[:, None], nbs]                  # (G,k)
            mask_valid = np.isfinite(dist_anchor_nb) & (dist_anchor_nb <= geodesic_radius)
            w = np.where(mask_valid, w, 0.0)

        w_sum = w.sum(axis=1)                                   # (G,)

        # --- weighted average of vectors from valid neighbors ---
        V_nb = Vector[nbs]                                           # (G,k,2)
        num = (V_nb * w[..., None]).sum(axis=1)                 # (G,2)
        den = np.maximum(w_sum, 1e-12)[:, None]                 # (G,1) guard against zero
        V_grid = num / den                                      # (G,2)

        thr = np.percentile(w_sum, grid_thresh) if 0 < grid_thresh < 100 else float(grid_thresh)
        keep = (w_sum >= thr) & (dis_anchor <= 0.5 * gridsize)
        grid_pts_plot, V_grid_plot = grid_pts[keep].copy(), V_grid[keep].copy()
        V_grid_adj = V_grid.copy()
        V_grid[~keep] = 0.0
    
    if group_cmap is not None:
        if group_cmap.lower() == 'plotly':
            group_cmap = plotly.colors.qualitative.Plotly
        elif group_cmap.lower() == 'light24':
            group_cmap = plotly.colors.qualitative.Light24
        elif group_cmap.lower() == 'dark24':
            group_cmap = plotly.colors.qualitative.Dark24
        elif group_cmap.lower() == 'alphabet':
            group_cmap = plotly.colors.qualitative.Alphabet

    idx = np.argsort(ndcolor)
    if background == 'summary' or background == 'group':
        if not ndsize==0:
            if background == 'summary':
                sc = ax.scatter(coords[idx,0], coords[idx,1], s=ndsize, c=ndcolor[idx], cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)
            elif background == 'group':
                labels = np.array( adata.obs[group_name], str )
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx = np.where(labels == unique_labels[i_label])[0]
                    ax.scatter(coords[idx,0], coords[idx,1], s=ndsize, c=group_cmap[unique_labels[i_label]], linewidth=0, label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0,0.0])
        if plot_method == "cell":
            scale = auto_quiver_scale(X_vec, V_cell, frac=largest_arrow)
            ax.quiver(X_vec[:,0], X_vec[:,1], V_cell[:,0], V_cell[:,1], scale=scale, angles='xy', scale_units='xy', width=arrow_width, color=arrow_color, pivot=quiver_pivot)
        elif plot_method == "grid":
            scale = auto_quiver_scale(grid_pts_plot, V_grid_plot, frac=largest_arrow)
            ax.quiver(grid_pts_plot[:,0], grid_pts_plot[:,1], V_grid_plot[:,0], V_grid_plot[:,1], scale=scale, angles='xy', scale_units='xy', width=arrow_width, color=arrow_color, pivot=quiver_pivot)
    
    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        if library_id is None:
            library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')
        if plot_method == "cell":
            scale = auto_quiver_scale(X_vec, V_cell, frac=largest_arrow)
            ax.quiver(X_vec[:,0]*sf, X_vec[:,1]*sf, V_cell[:,0]*sf, V_cell[:,1]*sf, scale=scale, angles='xy', scale_units='xy', width=arrow_width, color=arrow_color, pivot=quiver_pivot)
        elif plot_method == "grid":
            scale = auto_quiver_scale(grid_pts_plot, V_grid_plot, frac=largest_arrow)
            ax.quiver(grid_pts_plot[:,0]*sf, grid_pts_plot[:,1]*sf, V_grid_plot[:,0]*sf, V_grid_plot[:,1]*sf, scale=scale, angles='xy', scale_units='xy', width=arrow_width, color=arrow_color, pivot=quiver_pivot)
    
    ax.set_title(title)
    if background == 'summary':
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signal Strength", fontsize=10)
    ax.axis("equal")
    ax.axis("off")
    ax.invert_yaxis()
    
    if not plot_savepath is None:
        plt.savefig(plot_savepath, dpi=500, bbox_inches = 'tight', transparent=True)

    if plot_method == "cell":
        return ax, coords, V_cell
    elif plot_method == "grid":
        return ax, grid_pts, V_grid
    elif plot_method == None:
        return ax, None, None
