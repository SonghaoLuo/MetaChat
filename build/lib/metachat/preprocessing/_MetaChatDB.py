import io
import pkgutil
import pandas as pd

def MetaChatDB(
    species = "mouse"
):
    """
    Extract metabolite-sensor pairs from MetaChatDB.

    Parameters
    ----------
    species
        The species of the ligand-receptor pairs. Choose between 'mouse' and 'human'.

    Returns
    -------
    df_metasen : pandas.DataFrame
        A pandas DataFrame of the MS pairs with the six columns representing the Metabolite, Sensor, Metabolite.Pathway, Sensor.Pathway, Metabolite.Names, Long.Range.Channel respectively.

    """
    
    data = pkgutil.get_data(__name__, "_data/MetaChatDB/MetaChatDB_"+species+".csv")
    df_metasen = pd.read_csv(io.BytesIO(data), index_col=0)

    return df_metasen