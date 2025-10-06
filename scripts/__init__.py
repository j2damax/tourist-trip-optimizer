"""
Tourist Trip Optimizer - Core Modules

This package contains the core implementation of the TTDP solution.
"""

from .data_utils import (
    load_attractions_data,
    calculate_distance_matrix,
    prepare_data_for_optimization
)
from .ga_core import GeneticAlgorithm

__version__ = '1.0.0'
__all__ = [
    'load_attractions_data',
    'calculate_distance_matrix',
    'prepare_data_for_optimization',
    'GeneticAlgorithm'
]
