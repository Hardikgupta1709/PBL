__version__ = '1.0.0'
__author__ = 'Hardik'

try:
    from Core_Strategy.conservative_strategy import ConservativeSystem, download_data
    from Core_Strategy.alternative_validation import validate_low_winrate_strategy
    from Core_Strategy.strategy_validator import StrategyValidator
    
    __all__ = [
        'ConservativeSystem',
        'download_data',
        'validate_low_winrate_strategy',
        'StrategyValidator'
    ]
except ImportError:
    pass