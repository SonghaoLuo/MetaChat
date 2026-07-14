import numpy as np
import scipy.sparse as sp


def sparse_min_merge(rows, cols, vals, shape, dtype=np.float32):
    """Merge (possibly duplicated) COO triplets into one sparse matrix, keeping the
    min value per (row, col). Unlike scipy's default COO->CSR (which sums duplicates)
    or elementwise .minimum() (which treats absent entries as 0), entries that never
    appear in `rows/cols` stay absent rather than being treated as 0 or +inf.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=dtype)

    if len(rows) == 0:
        return sp.csr_matrix(shape, dtype=dtype)

    flat = rows * shape[1] + cols
    order = np.argsort(flat, kind='stable')
    flat, vals, rows, cols = flat[order], vals[order], rows[order], cols[order]

    boundaries = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    min_vals = np.minimum.reduceat(vals, boundaries)

    return sp.csr_matrix((min_vals, (rows[boundaries], cols[boundaries])), shape=shape)
