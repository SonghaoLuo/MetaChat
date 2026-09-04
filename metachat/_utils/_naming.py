"""Display-name helpers for HMDB accessions.

MetaChat keys everything by HMDB accession -- ``adata.var_names``, the
``obsm`` communication summary columns (``s-HMDB0000220-Abca1``) and the
``uns``/``obsp`` keys.  That is deliberate: accessions are stable and
unambiguous.  They are, however, unreadable on a figure.

The helpers here translate accessions to metabolite names *at draw time
only*.  Nothing in this module mutates an AnnData object or renames a
stored result.
"""

import re

import pandas as pd

from .._settings import settings, _LABEL_STYLES

# HMDB accessions are 'HMDB' + 7 digits in MetaChatDB / Compass metadata.
_HMDB_PATTERN = re.compile(r"HMDB\d{7}")

_DB_CACHE = {}


def _metachatdb_map():
    """HMDB.ID -> Metabolite.Name from both MetaChatDB species tables.

    The mouse and human tables agree on every shared accession, so their
    union is unambiguous and we do not need to know the species.
    """
    if "metachatdb" not in _DB_CACHE:
        from ..preprocessing._importData import MetaChatDB

        frames = []
        for species in ("mouse", "human"):
            try:
                frames.append(MetaChatDB(species=species)[["HMDB.ID", "Metabolite.Name"]])
            except Exception:
                continue
        if frames:
            df = pd.concat(frames).dropna().drop_duplicates(subset="HMDB.ID")
            _DB_CACHE["metachatdb"] = dict(zip(df["HMDB.ID"], df["Metabolite.Name"]))
        else:
            _DB_CACHE["metachatdb"] = {}
    return _DB_CACHE["metachatdb"]


def _compass_map():
    """HMDB ID -> metName from the Compass metabolite metadata (fallback only).

    Compass lists several names per accession, so this is a lower-priority
    source than MetaChatDB; we keep the first occurrence.
    """
    if "compass" not in _DB_CACHE:
        from ..preprocessing._importData import compass_annotation

        try:
            df = compass_annotation()
            df = df[["ID", "metName"]].dropna().drop_duplicates(subset="ID")
            _DB_CACHE["compass"] = dict(zip(df["ID"], df["metName"]))
        except Exception:
            _DB_CACHE["compass"] = {}
    return _DB_CACHE["compass"]


def is_hmdb_id(value) -> bool:
    """Whether `value` is exactly an HMDB accession (e.g. ``'HMDB0000220'``)."""
    return isinstance(value, str) and _HMDB_PATTERN.fullmatch(value) is not None


def build_hmdb_name_map(
    adata=None,
    species: str = None,
    use_database: bool = True,
):
    """
    Build an ``HMDB accession -> metabolite name`` lookup for display.

    Sources are consulted in decreasing order of specificity; earlier
    sources win:

    1. ``adata.uns['df_metasen_filtered']`` -- the metabolite-sensor table
       actually used for the communication inference on this object.
    2. ``MetaChatDB`` (``species`` if given, otherwise both species; the
       tables do not conflict).
    3. ``compass_annotation()`` -- covers Compass-derived metabolites that
       are absent from MetaChatDB.

    Parameters
    ----------
    adata : anndata.AnnData, optional
        Object whose ``uns`` may carry a metabolite-sensor table.
    species : str, optional
        ``'mouse'`` or ``'human'``.  Only narrows source 2; leaving it as
        None is safe.
    use_database : bool, default=True
        Whether to fall back on the packaged databases (sources 2 and 3).
        Set False to use only what is stored on ``adata``.

    Returns
    -------
    name_map : dict
        Accessions not present in any source are simply absent -- callers
        fall back to showing the accession itself.
    """
    name_map = {}

    if use_database:
        name_map.update(_compass_map())
        if species is None:
            name_map.update(_metachatdb_map())
        else:
            from ..preprocessing._importData import MetaChatDB

            df = MetaChatDB(species=species)[["HMDB.ID", "Metabolite.Name"]]
            df = df.dropna().drop_duplicates(subset="HMDB.ID")
            name_map.update(dict(zip(df["HMDB.ID"], df["Metabolite.Name"])))

    if adata is not None:
        df_metasen = adata.uns.get("df_metasen_filtered", None)
        if isinstance(df_metasen, pd.DataFrame) and {"HMDB.ID", "Metabolite.Name"} <= set(df_metasen.columns):
            df = df_metasen[["HMDB.ID", "Metabolite.Name"]].dropna().drop_duplicates(subset="HMDB.ID")
            name_map.update(dict(zip(df["HMDB.ID"], df["Metabolite.Name"])))

    return name_map


def prettify_label(
    label,
    name_map: dict = None,
    style: str = None,
    adata=None,
):
    """
    Replace every HMDB accession inside a string with its metabolite name.

    Works on any label shape used in MetaChat because it substitutes
    accessions in place rather than parsing a fixed format::

        'HMDB0000220'          -> 'Palmitic acid'
        's-HMDB0000220-Abca1'  -> 's-Palmitic acid-Abca1'
        'total-total'          -> 'total-total'

    Note that structural prefixes (``s-`` / ``r-``) and sensor gene names
    are left untouched.

    Parameters
    ----------
    label : str
        Any string; non-strings are returned unchanged.
    name_map : dict, optional
        Accession -> name lookup.  Built from `adata` / the packaged
        databases when omitted.
    style : str, {'name', 'hmdb', 'both'}, optional
        Defaults to ``mc.settings.metabolite_labels``.  ``'hmdb'`` returns
        the label untouched.
    adata : anndata.AnnData, optional
        Used to build `name_map` when it is not supplied.

    Returns
    -------
    str
    """
    if style is None:
        style = settings.metabolite_labels
    if style not in _LABEL_STYLES:
        raise ValueError(f"`style` must be one of {_LABEL_STYLES}, got {style!r}.")

    if style == "hmdb" or not isinstance(label, str):
        return label
    if not _HMDB_PATTERN.search(label):
        return label

    if name_map is None:
        name_map = build_hmdb_name_map(adata=adata)

    def _sub(match):
        hmdb = match.group(0)
        name = name_map.get(hmdb, None)
        if name is None:
            return hmdb
        return f"{name} ({hmdb})" if style == "both" else name

    return _HMDB_PATTERN.sub(_sub, label)


def prettify_labels(
    labels,
    name_map: dict = None,
    style: str = None,
    adata=None,
):
    """
    Vectorised :func:`prettify_label` over an iterable of labels.

    Builds the lookup once, so prefer this over a comprehension when
    relabelling a whole axis.

    Returns
    -------
    list of str
    """
    if style is None:
        style = settings.metabolite_labels
    if style == "hmdb":
        return list(labels)
    if name_map is None:
        name_map = build_hmdb_name_map(adata=adata)
    return [prettify_label(x, name_map=name_map, style=style) for x in labels]


def resolve_metabolite_name(
    query: str,
    name_map: dict = None,
    adata=None,
):
    """
    Normalise a user-supplied metabolite to its HMDB accession.

    Lets plotting functions accept either ``'HMDB0000220'`` or
    ``'Palmitic acid'`` on input while everything downstream keeps
    indexing by accession.

    Parameters
    ----------
    query : str
        An HMDB accession or a metabolite name (case-insensitive).
    name_map : dict, optional
        Accession -> name lookup; built on demand when omitted.
    adata : anndata.AnnData, optional
        Used to build `name_map` when it is not supplied.

    Returns
    -------
    str
        The accession.  Unrecognised input is returned unchanged so that
        the caller raises its own, more informative error.
    """
    if not isinstance(query, str) or _HMDB_PATTERN.fullmatch(query):
        return query

    if name_map is None:
        name_map = build_hmdb_name_map(adata=adata)

    reverse = {str(v).lower(): k for k, v in name_map.items()}
    return reverse.get(query.lower(), query)
