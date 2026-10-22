# __init__ file
import warnings
import sys
warnings.filterwarnings("ignore", category=UserWarning)

__version__ = "1.0.0.post1"

from ._settings import settings
from ._utils import build_hmdb_name_map, prettify_label, prettify_labels

from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl

sys.modules.update({
    f"{__name__}.pp": pp,
    f"{__name__}.tl": tl,
    f"{__name__}.pl": pl
})
__all__ = [
    "__version__",
    "pp", "tl", "pl",
    "settings", "build_hmdb_name_map", "prettify_label", "prettify_labels",
]