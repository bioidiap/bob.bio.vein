from .AlignedMatching import AlignedMatching

from .HessianAlignment import HessianAlignment
from .HessianHistMatchAligned import HessianHistMatchAligned
from .HistogramsMatch import HistogramsMatch
from .MiuraMatchAligned import MiuraMatchAligned
from .AnnotationMatch import AnnotationMatch
from .MatchTemplate import MatchTemplate
from .KeypointsMatcher import KeypointsMatcher
from .MiuraMatchFusion import MiuraMatchFusion

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
