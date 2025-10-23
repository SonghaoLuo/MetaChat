# __init__ file
import warnings
import sys
warnings.filterwarnings("ignore", category=UserWarning)

from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl

sys.modules.update({
    f"{__name__}.pp": pp,
    f"{__name__}.tl": tl,
    f"{__name__}.pl": pl
})
__all__ = ["pp", "tl", "pl"]