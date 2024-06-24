import pandas as pd
import pkgutil
import io


def MetaChatDB(
    species = "mouse"
):
    """
    Extract ligand-receptor pairs from LR database.

    Parameters
    ----------
    database
        The name of the ligand-receptor database. Use 'CellChat' for CellChatDB [Jin2021]_ of 'CellPhoneDB_v4.0' for CellPhoneDB_v4.0 [Efremova2020]_.
    species
        The species of the ligand-receptor pairs. Choose between 'mouse' and 'human'.
    heteromeric_delimiter
        The character to separate the heteromeric units of heteromeric ligands and receptors. 
        For example, if the heteromeric receptor (TGFbR1, TGFbR2) will be represented as 'TGFbR1_TGFbR2' if this parameter is set to '_'.
    signaling_type
        The type of signaling. Choose from 'Secreted Signaling', 'Cell-Cell Contact', and 'ECM-Receptor' for CellChatDB or 'Secreted Signaling' and 'Cell-Cell Contact' for CellPhoneDB_v4.0. 
        If None, all pairs in the database are returned.

    Returns
    -------
    df_ligrec : pandas.DataFrame
        A pandas DataFrame of the LR pairs with the three columns representing the ligand, receptor, and the signaling pathway name, respectively.

    References
    ----------

    .. [Jin2021] Jin, S., Guerrero-Juarez, C. F., Zhang, L., Chang, I., Ramos, R., Kuan, C. H., ... & Nie, Q. (2021). 
        Inference and analysis of cell-cell communication using CellChat. Nature communications, 12(1), 1-20.
    .. [Efremova2020] Efremova, M., Vento-Tormo, M., Teichmann, S. A., & Vento-Tormo, R. (2020). 
        CellPhoneDB: inferring cell–cell communication from combined expression of multi-subunit ligand–receptor complexes. Nature protocols, 15(4), 1484-1506.

    """
    
    data = pkgutil.get_data(__name__, "_data/MetaChatDB/MetaChatDB_"+species+".csv")
    df_metsen = pd.read_csv(io.BytesIO(data), index_col=0)

    return df_metsen
