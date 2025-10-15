def plot_cell_signaling(
    X,
    V,
    distance,                      # (N x N) cell-to-cell distance matrix used for barrier filtering
    signal_sum,
    cmap="coolwarm",
    group_cmap=None,
    arrow_color="tab:blue",
    plot_method="cell",            # "cell" | "grid" | "stream"
    background='summary',          # "summary" | "group" | "image"
    group_name=None,
    background_legend=False,
    library_id=None,
    adata=None,
    summary='sender',
    normalize_summary_quantile=0.995,
    ndsize=1,
    scale=1.0,                     # quiver/stream scale
    grid_density=1,
    grid_scale=1.0,                # bandwidth multiplier for Gaussian weights
    grid_thresh=1.0,               # threshold factor for masking (uses support/length)
    grid_width=0.005,
    stream_density=1.0,
    stream_linewidth=1,
    stream_cutoff_perc=5,
    vmin=None,
    vmax=None,
    title=None,
    plot_savepath=None,
    ax=None
):
    """
    X: (N,2) cell coordinates
    V: (N,2) cell vectors
    distance: (N,N) precomputed cell-to-cell distance matrix encoding barriers
    """
    # --- color background preprocessing (summary intensity clipping) ---
    ndcolor = signal_sum.copy()
    ndcolor_percentile = np.percentile(ndcolor, normalize_summary_quantile * 100)
    ndcolor[ndcolor > ndcolor_percentile] = ndcolor_percentile

    # --- clean zero vectors for cell plotting ---
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum == 0)[0], :] = np.nan

    # positions where arrows start for "cell" vs "receiver"
    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    # --- build grid if needed ---
    if plot_method in ("grid", "stream"):
        # rectangular grid covering X with a small padding
        xl, xr = np.min(X[:, 0]), np.max(X[:, 0])
        epsilon = 0.02 * (xr - xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:, 1]), np.max(X[:, 1])
        epsilon = 0.02 * (yr - yl); yl -= epsilon; yr += epsilon

        ngrid_x = int(50 * grid_density)
        gridsize = (xr - xl) / float(ngrid_x)
        ngrid_y = int((yr - yl) / gridsize)
        # ensure at least 2
        ngrid_x = max(2, ngrid_x)
        ngrid_y = max(2, ngrid_y)

        x_grid = np.linspace(xl, xr, ngrid_x)
        y_grid = np.linspace(yl, yr, ngrid_y)
        mg = np.meshgrid(x_grid, y_grid)
        grid_pts = np.concatenate((mg[0].reshape(-1, 1), mg[1].reshape(-1, 1)), axis=1)

        # --- barrier-aware radius neighborhood (radius = gridsize) ---
        # For each grid point:
        # 1) find cells within Euclidean radius r = gridsize
        # 2) choose anchor as the nearest cell among those neighbors
        # 3) barrier filtering: keep only neighbors with distance(anchor, neighbor) <= gridsize in the given "distance" matrix
        # 4) weighted aggregation: Gaussian weights over Euclidean distance from the grid point; average directions + sum magnitudes
        from sklearn.neighbors import NearestNeighbors
        from scipy.stats import norm

        radius = gridsize
        sigma = max(radius * grid_scale * 0.5, 1e-8)   # Gaussian kernel bandwidth

        nn_mdl = NearestNeighbors(algorithm='kd_tree')
        nn_mdl.fit(X)

        G = grid_pts.shape[0]
        V_grid = np.zeros((G, 2), dtype=float)
        support = np.zeros(G, dtype=float)

        # loop over grid points
        for g in range(G):
            gp = grid_pts[g]

            # Euclidean distances from grid point to all cells
            dists = np.linalg.norm(X - gp, axis=1)
            # neighbors within radius
            idx_g = np.where(dists <= radius)[0]

            # no neighbor -> zero vector (strict mode)
            if len(idx_g) == 0:
                V_grid[g] = 0.0
                support[g] = 0.0
                continue

            # anchor = closest cell to this grid point within the radius set
            anchor_local = idx_g[np.argmin(dists[idx_g])]
            # barrier filtering: keep neighbors connected to anchor within the same radius in the provided "distance" (cell-to-cell) matrix
            d_anchor = distance[anchor_local, idx_g]
            keep_mask = (d_anchor <= radius)
            idx_keep = idx_g[keep_mask]
            dis_keep = dists[idx_keep]

            # all filtered out -> zero vector
            if len(idx_keep) == 0:
                V_grid[g] = 0.0
                support[g] = 0.0
                continue

            # Gaussian weights on Euclidean distance from the grid point
            w = norm.pdf(dis_keep, scale=sigma)
            if not np.any(w > 0):
                w = np.ones_like(dis_keep)

            # --- Direct weighted vector sum (no direction/magnitude split) ---
            V_nb = V[idx_keep]                         # (k_keep, 2)
            # 'w' is the Gaussian distance weight from the grid point to each kept neighbor
            # Option 1: weighted SUM (preserve energy; directions can cancel if opposite)
            Vg = (V_nb * w[:, None]).sum(axis=0)       # (2,)

            # If you prefer a pure weighted AVERAGE instead, uncomment the next two lines:
            # wsum = w.sum()
            # Vg = (V_nb * w[:, None]).sum(axis=0) / max(wsum, 1e-12)

            V_grid[g] = Vg
            support[g] = float(np.linalg.norm(Vg))     # use resulting vector norm as support

        # reshape for stream if needed
        if plot_method == "stream":
            V_grid_2 = V_grid.T.reshape(2, ngrid_y, ngrid_x)    # (2, Ny, Nx)
            support_grid = support.reshape(ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid_2 ** 2).sum(0))              # (Ny, Nx)
            # build a mask: drop very small vectors AND low support
            # threshold based on grid_thresh (log-like) and support percentile
            vthr = np.clip(10 ** (grid_thresh - 6), None, (np.nanmax(vlen) or 1) * 0.9)
            cutoff = (vlen < vthr)
            sup_thr = np.percentile(support_grid[~np.isnan(support_grid)], stream_cutoff_perc) if np.any(~np.isnan(support_grid)) else 0.0
            cutoff |= (support_grid < sup_thr)
            # mask out low-confidence cells by setting NaN in U component (streamplot checks for NaN)
            V_grid_2[0][cutoff] = np.nan

    # --- categorical color maps if needed ---
    if cmap == 'Plotly':
        cmap = plotly.colors.qualitative.Plotly
    elif cmap == 'Light24':
        cmap = plotly.colors.qualitative.Light24
    elif cmap == 'Dark24':
        cmap = plotly.colors.qualitative.Dark24
    elif cmap == 'Alphabet':
        cmap = plotly.colors.qualitative.Alphabet

    # --- plotting ---
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    idx_sort = np.argsort(ndcolor)

    if background in ('summary', 'group'):
        if ndsize != 0:
            if background == 'summary':
                sc = ax.scatter(X[idx_sort, 0], X[idx_sort, 1], s=ndsize, c=ndcolor[idx_sort],
                                cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)
            elif background == 'group':
                labels = np.array(adata.obs[group_name], str)
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx_lab = np.where(labels == unique_labels[i_label])[0]
                    if group_cmap is None:
                        ax.scatter(X[idx_lab, 0], X[idx_lab, 1], s=ndsize,
                                   c=cmap[i_label], linewidth=0,
                                   label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                    else:
                        ax.scatter(X[idx_lab, 0], X[idx_lab, 1], s=ndsize,
                                   c=group_cmap[unique_labels[i_label]], linewidth=0,
                                   label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0, 0.0])

        if plot_method == "cell":
            # Use data units for arrow length fidelity
            ax.quiver(X_vec[:, 0], X_vec[:, 1], V_cell[:, 0], V_cell[:, 1],
                      scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            # Optional: mask low-support grid points before plotting
            sup = support
            if np.any(sup > 0):
                thr = grid_thresh * (np.percentile(sup, 99) / 100.0)
                keep = (sup > thr)
                grid_pts_plot = grid_pts[keep]
                V_grid_plot = V_grid[keep]
            else:
                grid_pts_plot = grid_pts
                V_grid_plot = V_grid

            ax.quiver(grid_pts_plot[:, 0], grid_pts_plot[:, 1],
                      V_grid_plot[:, 0], V_grid_plot[:, 1],
                      scale=scale, angles='xy', scale_units='xy',
                      width=grid_width, color=arrow_color)

        elif plot_method == "stream":
            # Adaptive linewidth by local length
            lengths = np.sqrt((V_grid_2 ** 2).sum(0))
            maxlen = np.nanmax(lengths) if np.any(~np.isnan(lengths)) else 1.0
            lw = stream_linewidth * 2 * lengths / maxlen
            ax.streamplot(x_grid, y_grid, V_grid_2[0], V_grid_2[1],
                          color=arrow_color, density=stream_density,
                          linewidth=lw)

    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        if library_id is None:
            library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')

        if plot_method == "cell":
            ax.quiver(X_vec[:, 0] * sf, X_vec[:, 1] * sf,
                      V_cell[:, 0] * sf, V_cell[:, 1] * sf,
                      scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            # same masking as summary branch
            sup = support
            if np.any(sup > 0):
                thr = grid_thresh * (np.percentile(sup, 99) / 100.0)
                keep = (sup > thr)
                grid_pts_plot = grid_pts[keep]
                V_grid_plot = V_grid[keep]
            else:
                grid_pts_plot = grid_pts
                V_grid_plot = V_grid

            ax.quiver(grid_pts_plot[:, 0] * sf, grid_pts_plot[:, 1] * sf,
                      V_grid_plot[:, 0] * sf, V_grid_plot[:, 1] * sf,
                      scale=scale, angles='xy', scale_units='xy',
                      width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid_2 ** 2).sum(0))
            maxlen = np.nanmax(lengths) if np.any(~np.isnan(lengths)) else 1.0
            lw = stream_linewidth * 2 * lengths / maxlen
            ax.streamplot(x_grid * sf, y_grid * sf, V_grid_2[0] * sf, V_grid_2[1] * sf,
                          color=arrow_color, density=stream_density, linewidth=lw)

    ax.set_title(title)
    if background == 'summary':
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signal Strength", fontsize=10)

    # Keep Cartesian orientation consistent with scatter/spatial plots
    # ax.invert_yaxis()  # enable if you need to match a flipped image coordinate

    ax.axis("equal")
    ax.axis("off")
    if plot_savepath is not None:
        import matplotlib.pyplot as plt
        plt.savefig(plot_savepath, dpi=500, bbox_inches='tight', transparent=True)
    return ax



def plot_cell_signaling_old(X,
    V,
    signal_sum,
    cmap = "coolwarm",
    group_cmap = None,
    arrow_color = "tab:blue",
    plot_method = "cell",
    background = 'summary',
    group_name = None,
    background_legend = False,
    library_id = None,
    adata = None,
    summary = 'sender',
    normalize_summary_quantile = 0.995,
    ndsize = 1,
    scale = 1.0,
    grid_density = 1,
    grid_knn = None,
    grid_scale = 1.0,
    grid_thresh = 1.0,
    grid_width = 0.005,
    stream_density = 1.0,
    stream_linewidth = 1,
    stream_cutoff_perc = 5,
    vmin = None,
    vmax = None,
    title = None,
    plot_savepath = None,
    ax = None
):
    ndcolor = signal_sum.copy()
    ndcolor_percentile = np.percentile(ndcolor, normalize_summary_quantile*100)
    ndcolor[ndcolor > ndcolor_percentile] = ndcolor_percentile
    
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan
    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    if plot_method == "grid" or plot_method == "stream":
        # Get a rectangular grid
        xl, xr = np.min(X[:,0]), np.max(X[:,0])
        epsilon = 0.02*(xr-xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:,1]), np.max(X[:,1])
        epsilon = 0.02*(yr-yl); yl -= epsilon; yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr-xl) / float(ngrid_x)
        ngrid_y = int((yr-yl)/gridsize)
        meshgrid = np.meshgrid(np.linspace(xl,xr,ngrid_x), np.linspace(yl,yr,ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1,1), meshgrid[1].reshape(-1,1)), axis=1)
    
        if grid_knn is None:
            grid_knn = int( X.shape[0] / 50 )
        nn_mdl = NearestNeighbors()
        nn_mdl.fit(X)
        dis, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        w = norm.pdf(x=dis, scale=gridsize * grid_scale)
        w_sum = w.sum(axis=1)

        V_grid = (V[nbs] * w[:,:,None]).sum(axis=1)
        V_grid /= np.maximum(1, w_sum)[:,None]

        if plot_method == "grid":
            grid_thresh *= np.percentile(w_sum, 99) / 100
            grid_pts, V_grid = grid_pts[w_sum > grid_thresh], V_grid[w_sum > grid_thresh]
        elif plot_method == "stream":
            x_grid = np.linspace(xl, xr, ngrid_x)
            y_grid = np.linspace(yl, yr, ngrid_y)
            V_grid = V_grid.T.reshape(2, ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid ** 2).sum(0))
            grid_thresh = 10 ** (grid_thresh - 6)
            grid_thresh = np.clip(grid_thresh, None, np.max(vlen) * 0.9)
            cutoff = vlen.reshape(V_grid[0].shape) < grid_thresh
            length = np.sum(np.mean(np.abs(V[nbs]),axis=1),axis=1).T
            length = length.reshape(ngrid_y, ngrid_x)
            cutoff |= length < np.percentile(length, stream_cutoff_perc)
            V_grid[0][cutoff] = np.nan

    if cmap == 'Plotly':
        cmap = plotly.colors.qualitative.Plotly
    elif cmap == 'Light24':
        cmap = plotly.colors.qualitative.Light24
    elif cmap == 'Dark24':
        cmap = plotly.colors.qualitative.Dark24
    elif cmap == 'Alphabet':
        cmap = plotly.colors.qualitative.Alphabet

    idx = np.argsort(ndcolor)
    if background == 'summary' or background == 'group':
        if not ndsize==0:
            if background == 'summary':
                sc = ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=ndcolor[idx], cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)
            elif background == 'group':
                labels = np.array( adata.obs[group_name], str )
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx = np.where(labels == unique_labels[i_label])[0]
                    if group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cmap[i_label], linewidth=0, label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                    elif not group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=group_cmap[unique_labels[i_label]], linewidth=0, label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0,0.0])
        if plot_method == "cell":
            ax.quiver(X_vec[:,0], X_vec[:,1], V_cell[:,0], -V_cell[:,1], scale=scale, scale_units='x', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0], grid_pts[:,1], V_grid[:,0], -V_grid[:,1], scale=scale, scale_units='x', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid, y_grid, V_grid[0], -V_grid[1], color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    
    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        if library_id is None:
            library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')
        if plot_method == "cell":
            ax.quiver(X_vec[:,0]*sf, X_vec[:,1]*sf, V_cell[:,0]*sf, -V_cell[:,1]*sf, scale=scale, scale_units='x', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0]*sf, grid_pts[:,1]*sf, V_grid[:,0]*sf, -V_grid[:,1]*sf, scale=scale, scale_units='x', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid*sf, y_grid*sf, V_grid[0]*sf, -V_grid[1]*sf, color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    ax.set_title(title)
    if background == 'summary':
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signal Strength", fontsize=10)
    # ax.invert_yaxis() doesn't change the arrow direction, so manually set the y component *(-1) in ax.quiver or ax.streamplot. 
    # This is to make the plot made consistent with sc.pl.spatial or sq.pl.spatial_scatter
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    if not plot_savepath is None:
        plt.savefig(plot_savepath, dpi=500, bbox_inches = 'tight', transparent=True)

def plot_cell_signaling_orginal(X,
    V,
    distance,
    signal_sum,
    cmap = "coolwarm",
    group_cmap = None,
    arrow_color = "tab:blue",
    plot_method = "cell",
    background = 'summary',
    group_name = None,
    background_legend = False,
    library_id = None,
    adata = None,
    summary = 'sender',
    normalize_summary_quantile = 0.995,
    ndsize = 1,
    scale = 1.0,
    grid_density = 1,
    grid_knn = None,
    grid_scale = 1.0,
    grid_thresh = 1.0,
    grid_width = 0.005,
    stream_density = 1.0,
    stream_linewidth = 1,
    stream_cutoff_perc = 5,
    vmin = None,
    vmax = None,
    title = None,
    plot_savepath = None,
    ax = None
):
    ndcolor = signal_sum.copy()
    ndcolor_percentile = np.percentile(ndcolor, normalize_summary_quantile*100)
    ndcolor[ndcolor > ndcolor_percentile] = ndcolor_percentile
    
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan
    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    if plot_method == "grid" or plot_method == "stream":
        # Get a rectangular grid
        xl, xr = np.min(X[:,0]), np.max(X[:,0])
        epsilon = 0.02*(xr-xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:,1]), np.max(X[:,1])
        epsilon = 0.02*(yr-yl); yl -= epsilon; yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr-xl) / float(ngrid_x)
        ngrid_y = int((yr-yl)/gridsize)
        meshgrid = np.meshgrid(np.linspace(xl,xr,ngrid_x), np.linspace(yl,yr,ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1,1), meshgrid[1].reshape(-1,1)), axis=1)
    
        # if grid_knn is None:
        #     grid_knn = int( X.shape[0] / 100 )
        # nn_mdl = NearestNeighbors()
        # nn_mdl.fit(X)
        # dis, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        # w = norm.pdf(x=dis, scale=gridsize * grid_scale)
        # w_sum = w.sum(axis=1)

        # V_grid = (V[nbs] * w[:,:,None]).sum(axis=1)
        # V_grid /= np.maximum(1, w_sum)[:,None]

        # --- Barrier-aware neighborhood for grid aggregation ---
        radius = gridsize                  # search radius around each grid point
        min_k  = 3                         # fallback K if empty after filtering
        sigma  = max(radius * 0.5, 1e-8)   # Gaussian kernel bandwidth

        # Build a KD-tree on cell coordinates for fast radius queries and nearest anchor lookup
        nn_mdl = NearestNeighbors(algorithm='kd_tree')
        nn_mdl.fit(X)

        # Pre-allocate outputs
        G = grid_pts.shape[0]
        V_grid = np.zeros((G, 2), dtype=float)
        support = np.zeros(G, dtype=float)   # optional: local energy for masking

        # For stream mode we will need a full rectangular grid later; here we just compute per-point vectors
        # 1) for each grid point: neighbors within radius (Euclidean), 2) anchor = nearest cell to grid point
        # 3) barrier filter: keep only neighbors whose cell-to-anchor distance in D_cc <= radius
        # 4) weighted aggregation on the remaining neighbors
        #    Robust scheme: average directions + sum magnitudes (preserves strength contrast)
        for g in range(G):
            # (a) cells within Euclidean radius around the grid point
            d_list, i_list = nn_mdl.radius_neighbors(
                grid_pts[g].reshape(1, -1), radius=radius, return_distance=True, sort_results=True
            )
            dis_g = d_list[0]   # shape (k_raw,)
            idx_g = i_list[0]   # shape (k_raw,)

            # (b) anchor cell: the nearest cell to the grid point (1-NN)
            d1, i1 = nn_mdl.kneighbors(grid_pts[g].reshape(1, -1), n_neighbors=1, return_distance=True)
            anchor = int(i1[0, 0])

            # (c) if no neighbor inside radius, fallback to a few nearest cells (for robustness)
            if len(idx_g) == 0:
                # take min_k nearest as a fallback neighborhood
                d_fbk, i_fbk = nn_mdl.kneighbors(grid_pts[g].reshape(1, -1), n_neighbors=min_k, return_distance=True)
                dis_g = d_fbk[0]
                idx_g = i_fbk[0]

            # (d) barrier filter using your precomputed cell-to-cell distance matrix D_cc
            #     Keep only neighbors whose distance to the anchor cell is <= radius
            d_anchor = distance[anchor, idx_g]                   # shape (k_use,)
            mask_barrier = (d_anchor <= radius)
            idx_keep = idx_g[mask_barrier]
            dis_keep = dis_g[mask_barrier]

            # If everything was filtered out, fall back to the anchor cell itself
            if len(idx_keep) == 0:
                idx_keep = np.array([anchor], dtype=int)
                # use the true Euclidean distance to anchor as weight distance
                dis_anchor = np.linalg.norm(X[anchor] - grid_pts[g])
                dis_keep = np.array([dis_anchor], dtype=float)

            # (e) distance weights (Gaussian on Euclidean distance from the grid point)
            w = norm.pdf(dis_keep, scale=sigma)              # shape (k_keep,)
            # If all weights are zero (rare), add a tiny epsilon to avoid zero-division
            if not np.any(w > 0):
                w = np.ones_like(w)

            # (f) robust aggregation: average directions, sum magnitudes
            V_nb   = V[idx_keep]                             # (k_keep, 2)
            mag_nb = np.linalg.norm(V_nb, axis=1)            # (k_keep,)
            dir_nb = V_nb / (mag_nb[:, None] + 1e-8)         # unit directions

            # weighted mean direction (normalize after summation)
            dir_w = (dir_nb * w[:, None]).sum(axis=0)
            dir_w /= (np.linalg.norm(dir_w) + 1e-8)

            # weighted SUM magnitude (preserves strength; do NOT divide by sum of weights)
            mag_w = (mag_nb * w).sum()

            # combine back to a vector at this grid point
            V_grid[g] = dir_w * mag_w

            # optional support score for later masking/threshold
            support[g] = (mag_nb * w).sum()


        if plot_method == "grid":
            grid_thresh *= np.percentile(w_sum, 99) / 100
            grid_pts, V_grid = grid_pts[w_sum > grid_thresh], V_grid[w_sum > grid_thresh]
        elif plot_method == "stream":
            x_grid = np.linspace(xl, xr, ngrid_x)
            y_grid = np.linspace(yl, yr, ngrid_y)
            V_grid = V_grid.T.reshape(2, ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid ** 2).sum(0))
            grid_thresh = 10 ** (grid_thresh - 6)
            grid_thresh = np.clip(grid_thresh, None, np.max(vlen) * 0.9)
            cutoff = vlen.reshape(V_grid[0].shape) < grid_thresh
            length = np.sum(np.mean(np.abs(V[nbs]),axis=1),axis=1).T
            length = length.reshape(ngrid_y, ngrid_x)
            cutoff |= length < np.percentile(length, stream_cutoff_perc)
            V_grid[0][cutoff] = np.nan

    if cmap == 'Plotly':
        cmap = plotly.colors.qualitative.Plotly
    elif cmap == 'Light24':
        cmap = plotly.colors.qualitative.Light24
    elif cmap == 'Dark24':
        cmap = plotly.colors.qualitative.Dark24
    elif cmap == 'Alphabet':
        cmap = plotly.colors.qualitative.Alphabet

    idx = np.argsort(ndcolor)
    if background == 'summary' or background == 'group':
        if not ndsize==0:
            if background == 'summary':
                sc = ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=ndcolor[idx], cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)
            elif background == 'group':
                labels = np.array( adata.obs[group_name], str )
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx = np.where(labels == unique_labels[i_label])[0]
                    if group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cmap[i_label], linewidth=0, label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                    elif not group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=group_cmap[unique_labels[i_label]], linewidth=0, label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0,0.0])
        if plot_method == "cell":
            ax.quiver(X_vec[:,0], X_vec[:,1], V_cell[:,0], V_cell[:,1], scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0], grid_pts[:,1], V_grid[:,0], V_grid[:,1], scale=scale, angles='xy', scale_units='xy', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid, y_grid, V_grid[0], V_grid[1], color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    
    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        if library_id is None:
            library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')
        if plot_method == "cell":
            ax.quiver(X_vec[:,0]*sf, X_vec[:,1]*sf, V_cell[:,0]*sf, V_cell[:,1]*sf, scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0]*sf, grid_pts[:,1]*sf, V_grid[:,0]*sf, V_grid[:,1]*sf, scale=scale, angles='xy', scale_units='xy', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid*sf, y_grid*sf, V_grid[0]*sf, V_grid[1]*sf, color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    ax.set_title(title)
    if background == 'summary':
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signal Strength", fontsize=10)
    # ax.invert_yaxis() doesn't change the arrow direction, so manually set the y component *(-1) in ax.quiver or ax.streamplot. 
    # This is to make the plot made consistent with sc.pl.spatial or sq.pl.spatial_scatter
    # ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    if not plot_savepath is None:
        plt.savefig(plot_savepath, dpi=500, bbox_inches = 'tight', transparent=True)

def plot_cell_signaling_orginal(X,
    V,
    distance,
    signal_sum,
    cmap = "coolwarm",
    group_cmap = None,
    arrow_color = "tab:blue",
    plot_method = "cell",
    background = 'summary',
    group_name = None,
    background_legend = False,
    library_id = None,
    adata = None,
    summary = 'sender',
    normalize_summary_quantile = 0.995,
    ndsize = 1,
    scale = 1.0,
    grid_density = 1,
    grid_knn = None,
    grid_scale = 1.0,
    grid_thresh = 1.0,
    grid_width = 0.005,
    stream_density = 1.0,
    stream_linewidth = 1,
    stream_cutoff_perc = 5,
    vmin = None,
    vmax = None,
    title = None,
    plot_savepath = None,
    ax = None
):
    ndcolor = signal_sum.copy()
    ndcolor_percentile = np.percentile(ndcolor, normalize_summary_quantile*100)
    ndcolor[ndcolor > ndcolor_percentile] = ndcolor_percentile
    
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan

    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    if plot_method == "grid" or plot_method == "stream":
        # Get a rectangular grid
        xl, xr = np.min(X[:,0]), np.max(X[:,0])
        epsilon = 0.02*(xr-xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:,1]), np.max(X[:,1])
        epsilon = 0.02*(yr-yl); yl -= epsilon; yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr-xl) / float(ngrid_x)
        ngrid_y = int((yr-yl)/gridsize)
        meshgrid = np.meshgrid(np.linspace(xl,xr,ngrid_x), np.linspace(yl,yr,ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1,1), meshgrid[1].reshape(-1,1)), axis=1)
    
        # if grid_knn is None:
        #     grid_knn = int( X.shape[0] / 100 )
        # nn_mdl = NearestNeighbors()
        # nn_mdl.fit(X)
        # dis, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        # w = norm.pdf(x=dis, scale=gridsize * grid_scale)
        # w_sum = w.sum(axis=1)

        # V_grid = (V[nbs] * w[:,:,None]).sum(axis=1)
        # V_grid /= np.maximum(1, w_sum)[:,None]

        # --- Barrier-aware neighborhood for grid aggregation ---
        radius = gridsize                  # search radius around each grid point
        min_k  = 3                         # fallback K if empty after filtering
        sigma  = max(radius * 0.5, 1e-8)   # Gaussian kernel bandwidth

        # Build a KD-tree on cell coordinates for fast radius queries and nearest anchor lookup
        nn_mdl = NearestNeighbors(algorithm='kd_tree')
        nn_mdl.fit(X)

        # Pre-allocate outputs
        G = grid_pts.shape[0]
        V_grid = np.zeros((G, 2), dtype=float)
        support = np.zeros(G, dtype=float)   # optional: local energy for masking

        # For stream mode we will need a full rectangular grid later; here we just compute per-point vectors
        # 1) for each grid point: neighbors within radius (Euclidean), 2) anchor = nearest cell to grid point
        # 3) barrier filter: keep only neighbors whose cell-to-anchor distance in D_cc <= radius
        # 4) weighted aggregation on the remaining neighbors
        #    Robust scheme: average directions + sum magnitudes (preserves strength contrast)
        for g in range(G):
            # (a) cells within Euclidean radius around the grid point
            d_list, i_list = nn_mdl.radius_neighbors(
                grid_pts[g].reshape(1, -1), radius=radius, return_distance=True, sort_results=True
            )
            dis_g = d_list[0]   # shape (k_raw,)
            idx_g = i_list[0]   # shape (k_raw,)

            # (b) anchor cell: the nearest cell to the grid point (1-NN)
            d1, i1 = nn_mdl.kneighbors(grid_pts[g].reshape(1, -1), n_neighbors=1, return_distance=True)
            anchor = int(i1[0, 0])

            # (c) if no neighbor inside radius, fallback to a few nearest cells (for robustness)
            if len(idx_g) == 0:
                # take min_k nearest as a fallback neighborhood
                d_fbk, i_fbk = nn_mdl.kneighbors(grid_pts[g].reshape(1, -1), n_neighbors=min_k, return_distance=True)
                dis_g = d_fbk[0]
                idx_g = i_fbk[0]

            # (d) barrier filter using your precomputed cell-to-cell distance matrix D_cc
            #     Keep only neighbors whose distance to the anchor cell is <= radius
            d_anchor = distance[anchor, idx_g]                   # shape (k_use,)
            mask_barrier = (d_anchor <= radius)
            idx_keep = idx_g[mask_barrier]
            dis_keep = dis_g[mask_barrier]

            # If everything was filtered out, fall back to the anchor cell itself
            if len(idx_keep) == 0:
                idx_keep = np.array([anchor], dtype=int)
                # use the true Euclidean distance to anchor as weight distance
                dis_anchor = np.linalg.norm(X[anchor] - grid_pts[g])
                dis_keep = np.array([dis_anchor], dtype=float)

            # (e) distance weights (Gaussian on Euclidean distance from the grid point)
            w = norm.pdf(dis_keep, scale=sigma)              # shape (k_keep,)
            # If all weights are zero (rare), add a tiny epsilon to avoid zero-division
            if not np.any(w > 0):
                w = np.ones_like(w)

            # (f) robust aggregation: average directions, sum magnitudes
            V_nb   = V[idx_keep]                             # (k_keep, 2)
            mag_nb = np.linalg.norm(V_nb, axis=1)            # (k_keep,)
            dir_nb = V_nb / (mag_nb[:, None] + 1e-8)         # unit directions

            # weighted mean direction (normalize after summation)
            dir_w = (dir_nb * w[:, None]).sum(axis=0)
            dir_w /= (np.linalg.norm(dir_w) + 1e-8)

            # weighted SUM magnitude (preserves strength; do NOT divide by sum of weights)
            mag_w = (mag_nb * w).sum()

            # combine back to a vector at this grid point
            V_grid[g] = dir_w * mag_w

            # optional support score for later masking/threshold
            support[g] = (mag_nb * w).sum()


        if plot_method == "grid":
            grid_thresh *= np.percentile(w_sum, 99) / 100
            grid_pts, V_grid = grid_pts[w_sum > grid_thresh], V_grid[w_sum > grid_thresh]
        elif plot_method == "stream":
            x_grid = np.linspace(xl, xr, ngrid_x)
            y_grid = np.linspace(yl, yr, ngrid_y)
            V_grid = V_grid.T.reshape(2, ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid ** 2).sum(0))
            grid_thresh = 10 ** (grid_thresh - 6)
            grid_thresh = np.clip(grid_thresh, None, np.max(vlen) * 0.9)
            cutoff = vlen.reshape(V_grid[0].shape) < grid_thresh
            length = np.sum(np.mean(np.abs(V[nbs]),axis=1),axis=1).T
            length = length.reshape(ngrid_y, ngrid_x)
            cutoff |= length < np.percentile(length, stream_cutoff_perc)
            V_grid[0][cutoff] = np.nan

    if cmap == 'Plotly':
        cmap = plotly.colors.qualitative.Plotly
    elif cmap == 'Light24':
        cmap = plotly.colors.qualitative.Light24
    elif cmap == 'Dark24':
        cmap = plotly.colors.qualitative.Dark24
    elif cmap == 'Alphabet':
        cmap = plotly.colors.qualitative.Alphabet

    idx = np.argsort(ndcolor)
    if background == 'summary' or background == 'group':
        if not ndsize==0:
            if background == 'summary':
                sc = ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=ndcolor[idx], cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)
            elif background == 'group':
                labels = np.array( adata.obs[group_name], str )
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx = np.where(labels == unique_labels[i_label])[0]
                    if group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cmap[i_label], linewidth=0, label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                    elif not group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=group_cmap[unique_labels[i_label]], linewidth=0, label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0,0.0])
        if plot_method == "cell":
            ax.quiver(X_vec[:,0], X_vec[:,1], V_cell[:,0], V_cell[:,1], scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0], grid_pts[:,1], V_grid[:,0], V_grid[:,1], scale=scale, angles='xy', scale_units='xy', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid, y_grid, V_grid[0], V_grid[1], color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    
    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        if library_id is None:
            library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')
        if plot_method == "cell":
            ax.quiver(X_vec[:,0]*sf, X_vec[:,1]*sf, V_cell[:,0]*sf, V_cell[:,1]*sf, scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0]*sf, grid_pts[:,1]*sf, V_grid[:,0]*sf, V_grid[:,1]*sf, scale=scale, angles='xy', scale_units='xy', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid*sf, y_grid*sf, V_grid[0]*sf, V_grid[1]*sf, color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    ax.set_title(title)
    if background == 'summary':
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signal Strength", fontsize=10)
    # ax.invert_yaxis() doesn't change the arrow direction, so manually set the y component *(-1) in ax.quiver or ax.streamplot. 
    # This is to make the plot made consistent with sc.pl.spatial or sq.pl.spatial_scatter
    # ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    if not plot_savepath is None:
        plt.savefig(plot_savepath, dpi=500, bbox_inches = 'tight', transparent=True)


























def plot_cell_signaling_v2(
    X,
    V,
    distance,                      # (N x N) cell-to-cell distance matrix used for barrier filtering
    signal_sum,
    cmap="coolwarm",
    group_cmap=None,
    arrow_color="tab:blue",
    plot_method="cell",            # "cell" | "grid" | "stream"
    background='summary',          # "summary" | "group" | "image"
    group_name=None,
    background_legend=False,
    library_id=None,
    adata=None,
    summary='sender',
    normalize_summary_quantile=0.995,
    ndsize=1,
    scale=1.0,                     # quiver/stream scale
    grid_density=1,
    grid_scale=1.0,                # bandwidth multiplier for Gaussian weights
    grid_thresh=1.0,               # threshold factor for masking (uses support/length)
    grid_width=0.005,
    stream_density=1.0,
    stream_linewidth=1,
    stream_cutoff_perc=5,
    vmin=None,
    vmax=None,
    title=None,
    plot_savepath=None,
    ax=None
):
    """
    X: (N,2) cell coordinates
    V: (N,2) cell vectors
    distance: (N,N) precomputed cell-to-cell distance matrix encoding barriers
    """
    # --- color background preprocessing (summary intensity clipping) ---
    ndcolor = signal_sum.copy()
    ndcolor_percentile = np.percentile(ndcolor, normalize_summary_quantile * 100)
    ndcolor[ndcolor > ndcolor_percentile] = ndcolor_percentile

    # --- clean zero vectors for cell plotting ---
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum == 0)[0], :] = np.nan

    # positions where arrows start for "cell" vs "receiver"
    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    # --- build grid if needed ---
    if plot_method in ("grid", "stream"):
        # rectangular grid covering X with a small padding
        xl, xr = np.min(X[:, 0]), np.max(X[:, 0])
        epsilon = 0.02 * (xr - xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:, 1]), np.max(X[:, 1])
        epsilon = 0.02 * (yr - yl); yl -= epsilon; yr += epsilon

        ngrid_x = int(50 * grid_density)
        gridsize = (xr - xl) / float(ngrid_x)
        ngrid_y = int((yr - yl) / gridsize)
        # ensure at least 2
        ngrid_x = max(2, ngrid_x)
        ngrid_y = max(2, ngrid_y)

        x_grid = np.linspace(xl, xr, ngrid_x)
        y_grid = np.linspace(yl, yr, ngrid_y)
        mg = np.meshgrid(x_grid, y_grid)
        grid_pts = np.concatenate((mg[0].reshape(-1, 1), mg[1].reshape(-1, 1)), axis=1)

        # --- barrier-aware radius neighborhood (radius = gridsize) ---
        # For each grid point:
        # 1) find cells within Euclidean radius r = gridsize
        # 2) choose anchor as the nearest cell among those neighbors
        # 3) barrier filtering: keep only neighbors with distance(anchor, neighbor) <= gridsize in the given "distance" matrix
        # 4) weighted aggregation: Gaussian weights over Euclidean distance from the grid point; average directions + sum magnitudes


        radius = 2 * gridsize
        sigma = max(radius * grid_scale * 0.5, 1e-8)   # Gaussian kernel bandwidth

        nn_mdl = NearestNeighbors(algorithm='kd_tree')
        nn_mdl.fit(X)

        G = grid_pts.shape[0]
        V_grid = np.zeros((G, 2), dtype=float)
        support = np.zeros(G, dtype=float)

        # loop over grid points
        for g in range(G):
            gp = grid_pts[g]

            # Euclidean distances from grid point to all cells
            dists = np.linalg.norm(X - gp, axis=1)
            # neighbors within radius
            idx_g = np.where(dists <= radius)[0]

            # no neighbor -> zero vector (strict mode)
            if len(idx_g) == 0:
                V_grid[g] = 0.0
                support[g] = 0.0
                continue

            # anchor = closest cell to this grid point within the radius set
            anchor_local = idx_g[np.argmin(dists[idx_g])]
            # barrier filtering: keep neighbors connected to anchor within the same radius in the provided "distance" (cell-to-cell) matrix
            d_anchor = distance[anchor_local, idx_g]
            keep_mask = (d_anchor <= radius)
            idx_keep = idx_g[keep_mask]
            dis_keep = dists[idx_keep]

            # all filtered out -> zero vector
            if len(idx_keep) == 0:
                V_grid[g] = 0.0
                support[g] = 0.0
                continue

            # Gaussian weights on Euclidean distance from the grid point
            w = norm.pdf(dis_keep, scale=sigma)
            if not np.any(w > 0):
                w = np.ones_like(dis_keep)
                
            # # --- Direct weighted vector sum (no direction/magnitude split) ---
            # V_nb = V[idx_keep]                         # (k_keep, 2)
            # # 'w' is the Gaussian distance weight from the grid point to each kept neighbor
            # # Option 1: weighted SUM (preserve energy; directions can cancel if opposite)
            # Vg = (V_nb * w[:, None]).sum(axis=0)       # (2,)

            # # If you prefer a pure weighted AVERAGE instead, uncomment the next two lines:
            # # wsum = w.sum()
            # # Vg = (V_nb * w[:, None]).sum(axis=0) / max(wsum, 1e-12)

            # V_grid[g] = Vg
            # support[g] = float(np.linalg.norm(Vg))     # use resulting vector norm as support

            # 已得到邻域索引 idx_keep 与权重 w
            V_nb   = V[idx_keep]                        # (k,2)
            mag_nb = np.linalg.norm(V_nb, axis=1)       # (k,)
            V_net  = (V_nb * w[:, None]).sum(axis=0)    # 净向量
            M_tot  = float((mag_nb * w).sum())          # 总强度（不受抵消）
            # 输出用于作图
            dir_net = V_net / (np.linalg.norm(V_net) + 1e-8)
            V_grid[g] = dir_net * M_tot                 # 用总量定长度，更“稳”
            support[g] = M_tot                          # 也可用作阈值/alpha

        # reshape for stream if needed
        if plot_method == "stream":
            V_grid_2 = V_grid.T.reshape(2, ngrid_y, ngrid_x)    # (2, Ny, Nx)
            support_grid = support.reshape(ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid_2 ** 2).sum(0))              # (Ny, Nx)
            # build a mask: drop very small vectors AND low support
            # threshold based on grid_thresh (log-like) and support percentile
            vthr = np.clip(10 ** (grid_thresh - 6), None, (np.nanmax(vlen) or 1) * 0.9)
            cutoff = (vlen < vthr)
            sup_thr = np.percentile(support_grid[~np.isnan(support_grid)], stream_cutoff_perc) if np.any(~np.isnan(support_grid)) else 0.0
            cutoff |= (support_grid < sup_thr)
            # mask out low-confidence cells by setting NaN in U component (streamplot checks for NaN)
            V_grid_2[0][cutoff] = np.nan

    # --- categorical color maps if needed ---
    if cmap == 'Plotly':
        cmap = plotly.colors.qualitative.Plotly
    elif cmap == 'Light24':
        cmap = plotly.colors.qualitative.Light24
    elif cmap == 'Dark24':
        cmap = plotly.colors.qualitative.Dark24
    elif cmap == 'Alphabet':
        cmap = plotly.colors.qualitative.Alphabet

    # --- plotting ---
    if ax is None:
        fig, ax = plt.subplots()

    idx_sort = np.argsort(ndcolor)

    if background in ('summary', 'group'):
        if ndsize != 0:
            if background == 'summary':
                sc = ax.scatter(X[idx_sort, 0], X[idx_sort, 1], s=ndsize, c=ndcolor[idx_sort],
                                cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)
            elif background == 'group':
                labels = np.array(adata.obs[group_name], str)
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx_lab = np.where(labels == unique_labels[i_label])[0]
                    if group_cmap is None:
                        ax.scatter(X[idx_lab, 0], X[idx_lab, 1], s=ndsize,
                                   c=cmap[i_label], linewidth=0,
                                   label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                    else:
                        ax.scatter(X[idx_lab, 0], X[idx_lab, 1], s=ndsize,
                                   c=group_cmap[unique_labels[i_label]], linewidth=0,
                                   label=unique_labels[i_label], vmin=vmin, vmax=vmax)
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0, 0.0])

        if plot_method == "cell":
            # Use data units for arrow length fidelity
            ax.quiver(X_vec[:, 0], X_vec[:, 1], V_cell[:, 0], V_cell[:, 1],
                      scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            # Optional: mask low-support grid points before plotting
            sup = support
            if np.any(sup > 0):
                thr = grid_thresh * (np.percentile(sup, 99) / 100.0)
                keep = (sup > thr)
                grid_pts_plot = grid_pts[keep]
                V_grid_plot = V_grid[keep]
            else:
                grid_pts_plot = grid_pts
                V_grid_plot = V_grid

            ax.quiver(grid_pts_plot[:, 0], grid_pts_plot[:, 1],
                      V_grid_plot[:, 0], V_grid_plot[:, 1],
                      scale=scale, angles='xy', scale_units='xy',
                      width=grid_width, color=arrow_color)

        elif plot_method == "stream":
            # Adaptive linewidth by local length
            lengths = np.sqrt((V_grid_2 ** 2).sum(0))
            maxlen = np.nanmax(lengths) if np.any(~np.isnan(lengths)) else 1.0
            lw = stream_linewidth * 2 * lengths / maxlen
            ax.streamplot(x_grid, y_grid, V_grid_2[0], V_grid_2[1],
                          color=arrow_color, density=stream_density,
                          linewidth=lw)

    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        if library_id is None:
            library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')

        if plot_method == "cell":
            ax.quiver(X_vec[:, 0] * sf, X_vec[:, 1] * sf,
                      V_cell[:, 0] * sf, V_cell[:, 1] * sf,
                      scale=scale, angles='xy', scale_units='xy', color=arrow_color)
        elif plot_method == "grid":
            # same masking as summary branch
            sup = support
            if np.any(sup > 0):
                thr = grid_thresh * (np.percentile(sup, 99) / 100.0)
                keep = (sup > thr)
                grid_pts_plot = grid_pts[keep]
                V_grid_plot = V_grid[keep]
            else:
                grid_pts_plot = grid_pts
                V_grid_plot = V_grid

            ax.quiver(grid_pts_plot[:, 0] * sf, grid_pts_plot[:, 1] * sf,
                      V_grid_plot[:, 0] * sf, V_grid_plot[:, 1] * sf,
                      scale=scale, angles='xy', scale_units='xy',
                      width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid_2 ** 2).sum(0))
            maxlen = np.nanmax(lengths) if np.any(~np.isnan(lengths)) else 1.0
            lw = stream_linewidth * 2 * lengths / maxlen
            ax.streamplot(x_grid * sf, y_grid * sf, V_grid_2[0] * sf, V_grid_2[1] * sf,
                          color=arrow_color, density=stream_density, linewidth=lw)

    ax.set_title(title)
    if background == 'summary':
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signal Strength", fontsize=10)

    # Keep Cartesian orientation consistent with scatter/spatial plots
    ax.invert_yaxis()  # enable if you need to match a flipped image coordinate

    ax.axis("equal")
    ax.axis("off")
    if plot_savepath is not None:
        plt.savefig(plot_savepath, dpi=500, bbox_inches='tight', transparent=True)
    return ax