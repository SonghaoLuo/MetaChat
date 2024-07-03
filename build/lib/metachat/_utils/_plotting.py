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

def plot_cell_signaling(X,
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
                ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=ndcolor[idx], cmap=cmap, linewidth=0)
            elif background == 'group':
                labels = np.array( adata.obs[group_name], str )
                unique_labels = np.sort(list(set(list(labels))))
                for i_label in range(len(unique_labels)):
                    idx = np.where(labels == unique_labels[i_label])[0]
                    if group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cmap[i_label], linewidth=0, label=unique_labels[i_label])
                    elif not group_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=group_cmap[unique_labels[i_label]], linewidth=0, label=unique_labels[i_label])
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
    # ax.invert_yaxis() doesn't change the arrow direction, so manually set the y component *(-1) in ax.quiver or ax.streamplot. 
    # This is to make the plot made consistent with sc.pl.spatial or sq.pl.spatial_scatter
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    if not plot_savepath is None:
        plt.savefig(plot_savepath, dpi=500, bbox_inches = 'tight', transparent=True)
