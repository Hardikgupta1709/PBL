"""
Core Strategy Package
Contains main trading strategies
"""

from .ultra_conservative_strategy import UltraConservativeSystem, download_data
from .alternative_validation import validate_low_winrate_strategy

__all__ = ['UltraConservativeSystem', 'download_data', 'validate_low_winrate_strategy']