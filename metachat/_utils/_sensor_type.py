"""Selection of metabolite-sensor pairs by the signaling mechanism of the sensor."""

import pandas as pd

# Canonical sensor-type selections. Each entry maps an option name to a predicate
# on the token set parsed from the 'Sensor.Type' column of MetaChatDB.
# Note that 'receptor', 'receptor_NR' and 'transporter' intentionally overlap on
# the dually annotated ('receptor,transporter') pairs: those sensors can act
# through either mechanism, so they are counted in both selections.
_SENSOR_TYPE_OPTIONS = {
    'all':              lambda t: True,
    'receptor':         lambda t: 'receptor' in t,
    'receptor_strict':  lambda t: t == {'receptor'},
    'receptor_NR':      lambda t: ('receptor' in t) or ('nuclear receptor' in t),
    'nuclear_receptor': lambda t: 'nuclear receptor' in t,
    'transporter':      lambda t: 'transporter' in t,
}

# Convenience spellings accepted from users, normalized to the canonical names above.
_SENSOR_TYPE_ALIASES = {
    'nr': 'nuclear_receptor',
    'receptor_nr': 'receptor_NR',
    'receptor+nr': 'receptor_NR',
    'nuclearreceptor': 'nuclear_receptor',
}

def _resolve_sensor_type(sensor_type):
    """Normalize a user-supplied sensor_type to a canonical option name."""
    if sensor_type is None:
        return 'all'
    key = str(sensor_type).strip().lower().replace(' ', '_').replace('-', '_')
    key = _SENSOR_TYPE_ALIASES.get(key, key).lower()
    # canonical names are matched case-insensitively ('receptor_NR' vs 'receptor_nr')
    for name in _SENSOR_TYPE_OPTIONS:
        if name.lower() == key:
            return name
    raise ValueError(
        f"Unknown sensor_type '{sensor_type}'. "
        f"Choose one of {list(_SENSOR_TYPE_OPTIONS.keys())}."
    )

def _sensor_type_suffix(sensor_type):
    """Key suffix used to keep results of different sensor types side by side.

    ``'all'`` yields an empty suffix so that existing results stay backward compatible.
    """
    sensor_type = _resolve_sensor_type(sensor_type)
    return '' if sensor_type == 'all' else '-' + sensor_type

def filter_sensor_type(
    df_metasen: pd.DataFrame,
    sensor_type: str = 'all'
):
    """
    Subset metabolite–sensor pairs by the signaling mechanism of the sensor.

    Parameters
    ----------
    df_metasen : pandas.DataFrame
        Metabolite–sensor pairs, e.g. from :func:`mc.pp.MetaChatDB` or
        ``adata.uns['df_metasen_filtered']``. Must contain a ``'Sensor.Type'`` column
        unless ``sensor_type='all'``.
    sensor_type : str, default='all'
        Which sensors to keep:

        - ``'all'`` : every sensor (default, identical to previous behavior).
        - ``'receptor'`` : membrane receptors, including dually annotated
          ``receptor,transporter`` sensors.
        - ``'receptor_strict'`` : membrane receptors only, excluding dually
          annotated sensors.
        - ``'receptor_NR'`` : ``'receptor'`` plus nuclear receptors.
        - ``'nuclear_receptor'`` (alias ``'NR'``) : nuclear receptors only.
        - ``'transporter'`` : transporters, including dually annotated sensors.

    Returns
    -------
    pandas.DataFrame
        The filtered pairs, with the index reset to a contiguous range.
    """
    sensor_type = _resolve_sensor_type(sensor_type)
    if sensor_type == 'all':
        return df_metasen.reset_index(drop=True).copy()

    if 'Sensor.Type' not in df_metasen.columns:
        raise KeyError(
            "Column 'Sensor.Type' is required to select sensors by type but was not found. "
            "Please use a version of MetaChatDB that provides sensor annotations, "
            "or keep sensor_type='all'."
        )

    keep = _SENSOR_TYPE_OPTIONS[sensor_type]
    mask = df_metasen['Sensor.Type'].apply(
        lambda v: keep({token.strip().lower() for token in str(v).split(',')})
    )
    df_out = df_metasen[mask].reset_index(drop=True).copy()
    if df_out.shape[0] == 0:
        raise ValueError(f"No metabolite-sensor pairs left after selecting sensor_type='{sensor_type}'.")
    return df_out

# Human-readable rendering of a selection, used in auto-generated plot titles.
_SENSOR_TYPE_LABELS = {
    'all': '',
    'receptor': 'receptor',
    'receptor_strict': 'receptor, strict',
    'receptor_NR': 'receptor + nuclear receptor',
    'nuclear_receptor': 'nuclear receptor',
    'transporter': 'transporter',
}

def _sensor_type_label(sensor_type):
    """Readable name of a sensor selection, or an empty string for ``'all'``."""
    return _SENSOR_TYPE_LABELS[_resolve_sensor_type(sensor_type)]
