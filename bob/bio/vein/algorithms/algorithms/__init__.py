from .AlgorithmBase import AlgorithmBase
from .HistogramsMatching import HistogramsMatching
from .SpatEnhancEigenvalMatching import SpatEnhancEigenvalMatching

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]