_LABEL_STYLES = ("name", "hmdb", "both")


class MetaChatSettings:
    """
    Global MetaChat settings.

    Attributes
    ----------
    metabolite_labels : str, {'name', 'hmdb', 'both'}, default='name'
        How HMDB accessions are rendered in plot titles, axis tick labels and
        legends.  ``'name'`` shows the metabolite name (e.g. ``Palmitic acid``),
        ``'hmdb'`` keeps the raw accession (e.g. ``HMDB0000220``) and ``'both'``
        shows ``Palmitic acid (HMDB0000220)``.

        This only affects what is drawn.  Stored results -- ``var_names``,
        ``obsm`` column names, ``uns`` keys and every returned DataFrame --
        always stay keyed by HMDB accession.
    """

    def __init__(self):
        self._metabolite_labels = "name"

    @property
    def metabolite_labels(self):
        return self._metabolite_labels

    @metabolite_labels.setter
    def metabolite_labels(self, value):
        if value not in _LABEL_STYLES:
            raise ValueError(
                f"`metabolite_labels` must be one of {_LABEL_STYLES}, got {value!r}."
            )
        self._metabolite_labels = value

    def __repr__(self):
        return f"MetaChatSettings(metabolite_labels={self._metabolite_labels!r})"


settings = MetaChatSettings()
