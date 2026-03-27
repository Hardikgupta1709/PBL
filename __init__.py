__version__ = '2.0.0'
__author__ = 'Hardik'

try:
    from Core_Strategy.conservative_strategy import ConservativeSystem, download_data
    from Core_Strategy.strategy_validator import StrategyValidator
    
    __all__ = [
        'ConservativeSystem',
        'download_data',
        'StrategyValidator'
    ]
except ImportError:
    pass