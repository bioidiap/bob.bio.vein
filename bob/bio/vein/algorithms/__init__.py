from .AlignedMatching import AlignedMatching

from .HessianAlignment import HessianAlignment
from .HessianHistMatchAligned import HessianHistMatchAligned
from .HistogramsMatch import HistogramsMatch
from .MiuraMatchAligned import MiuraMatchAligned
from .AnnotationMatch import AnnotationMatch
from .MatchTemplate import MatchTemplate

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
