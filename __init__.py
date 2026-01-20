"""
PBL_RUN - Pairs Trading System
================================

A production-ready pairs trading system combining:
- Kalman filtering for dynamic hedge ratios
- Machine learning regime classification
- Comprehensive validation framework
- Automated paper trading integration

Main Components:
- Core_Strategy: Trading strategy implementations
- Configuration: Validation and configuration tools
- Pair_Discovery: Automated pair finding and testing
- Paper_Trading: Alpaca integration for live/paper trading
- Testing: Unit tests and validation scripts
- Reports: Generated validation and performance reports

Author: Hardik
Version: 1.0
"""

__version__ = '1.0.0'
__author__ = 'Hardik'

# Make key classes available at package level
try:
    from Core_Strategy.ultra_conservative_strategy import UltraConservativeSystem, download_data
    from Core_Strategy.alternative_validation import validate_low_winrate_strategy
    from Core_Strategy.strategy_validator import StrategyValidator
    
    __all__ = [
        'UltraConservativeSystem',
        'download_data',
        'validate_low_winrate_strategy',
        'StrategyValidator'
    ]
except ImportError:
    # Graceful degradation if structure changes
    pass