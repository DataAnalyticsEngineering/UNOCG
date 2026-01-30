"""
Basic global data structures and utilties
"""

from enum import Enum

class Backend(Enum):
    TORCH = 1
    SCIPY = 2
    PETSC = 3
    FFTW = 4
