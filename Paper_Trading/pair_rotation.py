"""
Automatic Monthly Pair Rotation for Paper Trading
====================================================
Scans the expanded universe of pairs, ranks them by real-time
cointegration health, and selects the top-K healthiest pairs
for live paper trading.

Rotation Logic:
    1. Every rotation_interval_days (default: 30), re-scan all candidate pairs.
    2. Download trailing 1-year data for each pair.
    3. Compute rolling health score (ADF + Hurst + cointegration).
    4. Rank pairs by health_score; require is_healthy=True.
    5. Select top-K pairs (default: top 3).
    6. Update config.VALIDATED_PAIRS so the trader uses the new set.
    7. Close positions in any pairs that are being rotated out.

Pair Selection Criterion (from Research):
    - Pairs with < 5% healthy OOS days should be excluded entirely.
    - Only pairs with current health_score >= 0.5 are eligible.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from Core_Strategy.conservative_strategy import download_data, ConservativeSystem
from Research.dynamic_pair_selector import PairHealthMonitor, HealthThresholds

logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / 'Research' / 'results'
SUCCESS_PAIRS_CSV = RESULTS_DIR / 'success_pairs_full.csv'
EXPANDED_SUMMARY_CSV = RESULTS_DIR / 'expanded_universe_summary.csv'

# =============================================================================
# EXPANDED UNIVERSE — All discovered pairs from Research
# =============================================================================

EXPANDED_UNIVERSE = {
    # Banking
    'BAC_PNC': {'ticker_y': 'BAC', 'ticker_x': 'PNC', 'sector': 'Banking'},
    'USB_TFC': {'ticker_y': 'USB', 'ticker_x': 'TFC', 'sector': 'Banking'},
    'JPM_BAC': {'ticker_y': 'JPM', 'ticker_x': 'BAC', 'sector': 'Banking'},
    'GS_MS':   {'ticker_y': 'GS',  'ticker_x': 'MS',  'sector': 'Banking'},
    'C_WFC':   {'ticker_y': 'C',   'ticker_x': 'WFC', 'sector': 'Banking'},

    # Materials
    'LIN_VMC': {'ticker_y': 'LIN', 'ticker_x': 'VMC', 'sector': 'Materials'},
    'VMC_MLM': {'ticker_y': 'VMC', 'ticker_x': 'MLM', 'sector': 'Materials'},
    'LIN_APD': {'ticker_y': 'LIN', 'ticker_x': 'APD', 'sector': 'Materials'},
    'FCX_NEM': {'ticker_y': 'FCX', 'ticker_x': 'NEM', 'sector': 'Materials'},

    # Tech
    'GOOGL_AMD':  {'ticker_y': 'GOOGL', 'ticker_x': 'AMD',  'sector': 'Tech'},
    'MSFT_ORCL':  {'ticker_y': 'MSFT',  'ticker_x': 'ORCL', 'sector': 'Tech'},
    'NVDA_AMD':   {'ticker_y': 'NVDA',  'ticker_x': 'AMD',  'sector': 'Tech'},
    'INTC_AMD':   {'ticker_y': 'INTC',  'ticker_x': 'AMD',  'sector': 'Tech'},

    # Healthcare
    'JNJ_UNH':  {'ticker_y': 'JNJ',  'ticker_x': 'UNH',  'sector': 'Healthcare'},
    'PFE_MRK':  {'ticker_y': 'PFE',  'ticker_x': 'MRK',  'sector': 'Healthcare'},
    'ABBV_LLY': {'ticker_y': 'ABBV', 'ticker_x': 'LLY',  'sector': 'Healthcare'},
    'ABT_TMO':  {'ticker_y': 'ABT',  'ticker_x': 'TMO',  'sector': 'Healthcare'},

    # Utilities
    'AEP_SRE':  {'ticker_y': 'AEP', 'ticker_x': 'SRE', 'sector': 'Utilities'},
    'NEE_DUK':  {'ticker_y': 'NEE', 'ticker_x': 'DUK', 'sector': 'Utilities'},
    'SO_D':     {'ticker_y': 'SO',  'ticker_x': 'D',   'sector': 'Utilities'},
    'EXC_XEL':  {'ticker_y': 'EXC', 'ticker_x': 'XEL', 'sector': 'Utilities'},

    # Consumer
    'KO_COST':  {'ticker_y': 'KO',  'ticker_x': 'COST', 'sector': 'Consumer'},
    'KO_PEP':   {'ticker_y': 'KO',  'ticker_x': 'PEP',  'sector': 'Consumer'},
    'WMT_TGT':  {'ticker_y': 'WMT', 'ticker_x': 'TGT',  'sector': 'Consumer'},
    'MCD_SBUX': {'ticker_y': 'MCD', 'ticker_x': 'SBUX', 'sector': 'Consumer'},

    # Energy
    'CVX_COP':  {'ticker_y': 'CVX', 'ticker_x': 'COP', 'sector': 'Energy'},
    'XOM_CVX':  {'ticker_y': 'XOM', 'ticker_x': 'CVX', 'sector': 'Energy'},
    'EOG_OXY':  {'ticker_y': 'EOG', 'ticker_x': 'OXY', 'sector': 'Energy'},
    'SLB_HAL':  {'ticker_y': 'SLB', 'ticker_x': 'HAL', 'sector': 'Energy'},

    # Industrials
    'HON_DE':   {'ticker_y': 'HON', 'ticker_x': 'DE',  'sector': 'Industrials'},
    'CAT_DE':   {'ticker_y': 'CAT', 'ticker_x': 'DE',  'sector': 'Industrials'},
    'UPS_GE':   {'ticker_y': 'UPS', 'ticker_x': 'GE',  'sector': 'Industrials'},
    'BA_LMT':   {'ticker_y': 'BA',  'ticker_x': 'LMT', 'sector': 'Industrials'},

    # Telecom
    'T_VZ':     {'ticker_y': 'T',    'ticker_x': 'VZ',   'sector': 'Telecom'},
    'TMUS_CHTR':{'ticker_y': 'TMUS', 'ticker_x': 'CHTR', 'sector': 'Telecom'},

    # Retail
    'HD_LOW':   {'ticker_y': 'HD',   'ticker_x': 'LOW',  'sector': 'Retail'},
    'TJX_ROST': {'ticker_y': 'TJX',  'ticker_x': 'ROST', 'sector': 'Retail'},
    'DG_DLTR':  {'ticker_y': 'DG',   'ticker_x': 'DLTR', 'sector': 'Retail'},
}

# Default trading parameters (same for all pairs — no per-pair optimisation)
DEFAULT_PAIR_PARAMS = {
    'entry_z_normal':   1.5,
    'exit_z_normal':    0.4,
    'entry_z_volatile': 2.0,
    'exit_z_volatile':  0.25,
    'min_hold_days':    2,
    'z_score_window':   40,
    'position_size':    0.15,
    # Placeholder backtest stats (updated after health scan)
    'backtest_sharpe':  0.0,
    'backtest_return':  0.0,
    'expected_trades_year': 15,
    'win_rate':         50.0,
    'profit_factor':    1.0,
}


@dataclass
class RotationResult:
    """Result of a pair rotation scan."""
    timestamp: str
    pairs_scanned: int
    pairs_healthy: int
    selected_pairs: List[str]
    rotated_out: List[str]
    rotated_in: List[str]
    health_scores: Dict[str, float]
    rankings: List[Tuple[str, float, bool]]


class PairRotationManager:
    """
    Manages automatic monthly pair rotation for paper trading.

    Flow:
        1. scan_universe()    — Download data + compute health for all pairs
        2. rank_pairs()       — Rank by health score
        3. select_top_pairs() — Pick top-K healthy pairs
        4. apply_rotation()   — Update config and return rotation actions
    """

    def __init__(
        self,
        top_k: int = 3,
        min_health_score: float = 0.50,
        min_healthy_pct: float = 7.0,    # Day-10 aligned floor
        rotation_interval_days: int = 30,
        lookback_days: int = 365,
        health_window: int = 126,
        adf_pvalue: float = 0.05,
        hurst_max: float = 0.50,
        coint_pvalue: float = 0.10,
        max_total_exposure: float = 0.55,
        universe_mode: str = 'success',
        state_file: Optional[str] = None,
    ):
        self.top_k = top_k
        self.min_health_score = min_health_score
        self.min_healthy_pct = min_healthy_pct
        self.rotation_interval_days = rotation_interval_days
        self.lookback_days = lookback_days
        self.health_window = health_window
        self.max_total_exposure = max_total_exposure
        self.universe_mode = universe_mode

        # Dynamic universe source of truth from research outputs
        self.universe = self._load_research_universe(universe_mode)

        self.state_file = Path(state_file) if state_file else (
            BASE_DIR / 'Paper_Trading' / 'logs' / 'rotation_state.json'
        )
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.thresholds = HealthThresholds(
            adf_pvalue=adf_pvalue,
            hurst_max=hurst_max,
            coint_pvalue=coint_pvalue,
            lookback_window=health_window,
        )
        self.monitor = PairHealthMonitor(self.thresholds)

        # Current state
        self.current_pairs: List[str] = []
        self.last_rotation_date: Optional[str] = None
        self._load_state()

    def _load_state(self):
        """Load rotation state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.current_pairs = state.get('current_pairs', [])
                self.last_rotation_date = state.get('last_rotation_date', None)
                logger.info(f"Loaded rotation state: {len(self.current_pairs)} active pairs, "
                            f"last rotation: {self.last_rotation_date}")
            except Exception as e:
                logger.warning(f"Could not load rotation state: {e}")

    def _save_state(self):
        """Save rotation state to disk."""
        state = {
            'current_pairs': self.current_pairs,
            'last_rotation_date': self.last_rotation_date,
            'top_k': self.top_k,
            'min_health_score': self.min_health_score,
            'min_healthy_pct': self.min_healthy_pct,
            'universe_mode': self.universe_mode,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved rotation state to {self.state_file}")

    def _load_research_universe(self, mode: str = 'success') -> Dict[str, Dict]:
        """Load candidate universe from research results; fallback to static set."""
        sector_map = {}
        if EXPANDED_SUMMARY_CSV.exists():
            try:
                expanded = pd.read_csv(EXPANDED_SUMMARY_CSV)
                for _, row in expanded.iterrows():
                    sector_map[str(row['Pair'])] = str(row.get('Sector', 'Unknown'))
            except Exception as e:
                logger.warning(f"Could not load expanded universe summary: {e}")

        if mode == 'success' and SUCCESS_PAIRS_CSV.exists():
            try:
                success = pd.read_csv(SUCCESS_PAIRS_CSV)
                universe = {}
                for _, row in success.iterrows():
                    pair_slash = str(row['Pair'])
                    ty, tx = pair_slash.split('/')
                    pair_name = f"{ty}_{tx}"
                    universe[pair_name] = {
                        'ticker_y': ty,
                        'ticker_x': tx,
                        'sector': sector_map.get(pair_name, 'Unknown'),
                        'research_f_sharpe': float(row.get('F_Sharpe', 0.0)),
                        'research_pct_healthy': float(row.get('Pct_Healthy', 0.0)),
                    }
                if universe:
                    logger.info(f"Loaded {len(universe)} success pairs from {SUCCESS_PAIRS_CSV}")
                    return universe
            except Exception as e:
                logger.warning(f"Could not load success pairs CSV: {e}")

        if EXPANDED_SUMMARY_CSV.exists():
            try:
                expanded = pd.read_csv(EXPANDED_SUMMARY_CSV)
                universe = {}
                for _, row in expanded.iterrows():
                    pair_name = str(row['Pair'])
                    ty, tx = pair_name.split('_')
                    universe[pair_name] = {
                        'ticker_y': ty,
                        'ticker_x': tx,
                        'sector': str(row.get('Sector', 'Unknown')),
                    }
                if universe:
                    logger.info(f"Loaded {len(universe)} expanded pairs from {EXPANDED_SUMMARY_CSV}")
                    return universe
            except Exception as e:
                logger.warning(f"Could not load expanded universe CSV: {e}")

        logger.warning("Falling back to static EXPANDED_UNIVERSE")
        return EXPANDED_UNIVERSE

    def needs_rotation(self) -> bool:
        """Check if rotation is due based on interval."""
        if self.last_rotation_date is None:
            return True
        last = datetime.strptime(self.last_rotation_date, '%Y-%m-%d')
        days_since = (datetime.now() - last).days
        return days_since >= self.rotation_interval_days

    def scan_universe(
        self,
        universe: Optional[Dict] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """
        Download data and compute current health for all pairs in the universe.

        Returns dict mapping pair_name -> {
            'health_score': float,
            'is_healthy': bool,
            'checks_passed': int,
            'pct_healthy_recent': float,  # % healthy in last 63 days
            'adf_pvalue': float,
            'hurst': float,
            'coint_pvalue': float,
            'sector': str,
        }
        """
        if universe is None:
            universe = self.universe

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        results = {}
        failed = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"  PAIR UNIVERSE HEALTH SCAN — {end_str}")
            print(f"  Scanning {len(universe)} pairs across "
                  f"{len(set(v['sector'] for v in universe.values()))} sectors")
            print(f"{'='*70}")

        for pair_name, pair_info in universe.items():
            ticker_y = pair_info['ticker_y']
            ticker_x = pair_info['ticker_x']
            sector = pair_info.get('sector', 'Unknown')

            try:
                if verbose:
                    print(f"\n  Scanning {pair_name} ({ticker_y}/{ticker_x}, {sector})...", end=" ")

                # Download data
                stock_y, stock_x, market = download_data(
                    ticker_y, ticker_x, 'SPY', start_str, end_str
                )

                if len(stock_y) < self.health_window + 20:
                    if verbose:
                        print(f"SKIP (insufficient data: {len(stock_y)} pts)")
                    failed.append(pair_name)
                    continue

                # Compute spread using OLS hedge ratio on trailing data
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(stock_x.values.reshape(-1, 1), stock_y.values)
                hedge_ratio = model.coef_[0]
                spread = stock_y - hedge_ratio * stock_x

                # Compute rolling health
                health_df = self.monitor.compute_rolling_health(
                    stock_y, stock_x, spread, step=5
                )

                # Get latest health
                latest = health_df.iloc[-1]

                # Compute % healthy over recent quarter (63 trading days)
                recent_window = min(63, len(health_df))
                recent_health = health_df.tail(recent_window)
                pct_healthy_recent = recent_health['is_healthy'].mean() * 100

                results[pair_name] = {
                    'health_score': float(latest['health_score']),
                    'is_healthy': bool(latest['is_healthy']),
                    'checks_passed': int(latest['checks_passed']),
                    'pct_healthy_recent': pct_healthy_recent,
                    'adf_pvalue': float(latest.get('adf_pvalue', np.nan)),
                    'hurst': float(latest.get('hurst', np.nan)),
                    'coint_pvalue': float(latest.get('coint_pvalue', np.nan)),
                    'half_life': float(latest.get('half_life', np.nan)),
                    'hedge_ratio': float(hedge_ratio),
                    'sector': sector,
                    'data_points': len(stock_y),
                }

                if verbose:
                    status = "HEALTHY" if latest['is_healthy'] else "UNHEALTHY"
                    print(f"{status} (score={latest['health_score']:.3f}, "
                          f"recent={pct_healthy_recent:.0f}%)")

            except Exception as e:
                if verbose:
                    print(f"FAILED ({e})")
                failed.append(pair_name)
                logger.warning(f"Failed to scan {pair_name}: {e}")

        if verbose:
            print(f"\n  Scanned: {len(results)}/{len(universe)} pairs")
            if failed:
                print(f"  Failed: {', '.join(failed)}")

        return results

    def rank_pairs(self, scan_results: Dict[str, Dict]) -> List[Tuple[str, float, bool]]:
        """
        Rank pairs by health score (descending).

        Returns list of (pair_name, health_score, is_eligible).
        A pair is eligible if:
            - Currently healthy (is_healthy=True)
            - health_score >= min_health_score
            - Recent healthy % >= min_healthy_pct
        """
        rankings = []
        for pair_name, info in scan_results.items():
            eligible = (
                info['is_healthy'] and
                info['health_score'] >= self.min_health_score and
                info['pct_healthy_recent'] >= self.min_healthy_pct
            )
            rankings.append((pair_name, info['health_score'], eligible))

        # Sort by: eligible first (True > False), then by health_score desc
        rankings.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return rankings

    def select_top_pairs(
        self,
        rankings: List[Tuple[str, float, bool]],
        scan_results: Dict[str, Dict],
    ) -> List[str]:
        """Select top-K eligible pairs for trading."""
        eligible = [name for name, score, elig in rankings if elig]
        selected = eligible[:self.top_k]
        return selected

    def apply_rotation(
        self,
        selected_pairs: List[str],
        scan_results: Dict[str, Dict],
        verbose: bool = True,
    ) -> RotationResult:
        """
        Apply the rotation: update current_pairs and generate config entries.

        Returns RotationResult with details of what changed.
        """
        old_pairs = set(self.current_pairs)
        new_pairs = set(selected_pairs)

        rotated_out = list(old_pairs - new_pairs)
        rotated_in = list(new_pairs - old_pairs)

        self.current_pairs = selected_pairs
        self.last_rotation_date = datetime.now().strftime('%Y-%m-%d')
        self._save_state()

        # Build health scores dict
        health_scores = {
            name: scan_results[name]['health_score']
            for name in selected_pairs
            if name in scan_results
        }

        rankings = self.rank_pairs(scan_results)

        result = RotationResult(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            pairs_scanned=len(scan_results),
            pairs_healthy=sum(1 for _, _, e in rankings if e),
            selected_pairs=selected_pairs,
            rotated_out=rotated_out,
            rotated_in=rotated_in,
            health_scores=health_scores,
            rankings=rankings,
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"  ROTATION RESULT — {result.timestamp}")
            print(f"{'='*70}")
            print(f"  Pairs scanned:  {result.pairs_scanned}")
            print(f"  Pairs healthy:  {result.pairs_healthy}")
            print(f"  Selected (top-{self.top_k}): {', '.join(selected_pairs) or 'NONE'}")
            if rotated_out:
                print(f"  Rotated OUT: {', '.join(rotated_out)}")
            if rotated_in:
                print(f"  Rotated IN:  {', '.join(rotated_in)}")
            if not rotated_out and not rotated_in:
                print(f"  No changes — same pairs as before.")

            print(f"\n  Full Rankings:")
            print(f"  {'Rank':<6} {'Pair':<15} {'Score':>8} {'Eligible':>10} {'Status':>10}")
            print(f"  {'─'*51}")
            for i, (name, score, elig) in enumerate(rankings, 1):
                status = "ACTIVE" if name in new_pairs else ("eligible" if elig else "excluded")
                marker = " ←" if name in new_pairs else ""
                print(f"  {i:<6} {name:<15} {score:>8.3f} {'Yes' if elig else 'No':>10} {status:>10}{marker}")

        return result

    def build_validated_pairs_config(
        self,
        selected_pairs: List[str],
        scan_results: Dict[str, Dict],
    ) -> Dict:
        """
        Build a VALIDATED_PAIRS-compatible dict for the selected pairs.

        This can be written directly to config or used by the trader.
        """
        config = {}
        n_pairs = len(selected_pairs)

        # Distribute position size equally, capped at 60% total
        max_total_exposure = self.max_total_exposure
        per_pair_size = min(0.20, max_total_exposure / max(n_pairs, 1))

        for pair_name in selected_pairs:
            info = scan_results.get(pair_name, {})
            universe_info = self.universe.get(pair_name, EXPANDED_UNIVERSE.get(pair_name, {}))

            config[pair_name] = {
                'ticker_y': universe_info.get('ticker_y', pair_name.split('_')[0]),
                'ticker_x': universe_info.get('ticker_x', pair_name.split('_')[1]),
                **DEFAULT_PAIR_PARAMS,
                'position_size': round(per_pair_size, 2),
                'health_score': info.get('health_score', 0),
                'sector': info.get('sector', 'Unknown'),
                'backtest_sharpe': round(universe_info.get('research_f_sharpe', 0.0), 3),
                'win_rate': 50.0,
            }

        return config

    def run_rotation(
        self,
        universe: Optional[Dict] = None,
        force: bool = False,
        verbose: bool = True,
    ) -> Optional[RotationResult]:
        """
        Full rotation pipeline: scan → rank → select → apply.

        Parameters
        ----------
        universe : dict, optional
            Custom universe. Defaults to EXPANDED_UNIVERSE.
        force : bool
            Force rotation even if interval hasn't elapsed.
        verbose : bool
            Print progress.

        Returns
        -------
        RotationResult or None if rotation not needed.
        """
        if not force and not self.needs_rotation():
            days_since = (datetime.now() - datetime.strptime(
                self.last_rotation_date, '%Y-%m-%d')).days
            remaining = self.rotation_interval_days - days_since
            if verbose:
                print(f"\n  Rotation not due yet ({remaining} days remaining).")
                print(f"  Last rotation: {self.last_rotation_date}")
                print(f"  Active pairs: {', '.join(self.current_pairs)}")
            return None

        if verbose:
            print(f"\n{'#'*70}")
            print(f"  AUTOMATIC PAIR ROTATION")
            print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Interval: every {self.rotation_interval_days} days")
            print(f"  Top-K: {self.top_k} pairs")
            print(f"{'#'*70}")

        # Step 1: Scan universe
        scan_results = self.scan_universe(universe, verbose=verbose)

        if not scan_results:
            logger.error("No pairs could be scanned! Keeping current pairs.")
            return None

        # Step 2: Rank
        rankings = self.rank_pairs(scan_results)

        # Step 3: Select
        selected = self.select_top_pairs(rankings, scan_results)

        if not selected:
            # Fallback: keep risk filters, relax only score threshold
            relaxed = [
                name for name, score, _ in rankings
                if scan_results[name]['is_healthy']
                and scan_results[name]['pct_healthy_recent'] >= self.min_healthy_pct
            ][:self.top_k]

            if relaxed:
                selected = relaxed
                logger.warning(
                    "No pairs met strict score cutoff; using relaxed fallback among healthy candidates."
                )
                if verbose:
                    print("\n  WARNING: No pairs pass strict score threshold.")
                    print(f"  Fallback selected: {', '.join(selected)}")
            else:
                logger.warning("No eligible pairs found! Keeping current pairs.")
                if verbose:
                    print("\n  WARNING: No pairs pass health criteria!")
                    print(f"  Keeping current pairs: {', '.join(self.current_pairs)}")
                return None

        # Step 4: Apply
        result = self.apply_rotation(selected, scan_results, verbose=verbose)

        # Step 5: Build and save config
        new_config = self.build_validated_pairs_config(selected, scan_results)

        # Save rotation log
        log_file = BASE_DIR / 'Paper_Trading' / 'logs' / 'rotation_log.csv'
        log_entry = {
            'timestamp': result.timestamp,
            'pairs_scanned': result.pairs_scanned,
            'pairs_healthy': result.pairs_healthy,
            'selected': '|'.join(result.selected_pairs),
            'rotated_out': '|'.join(result.rotated_out),
            'rotated_in': '|'.join(result.rotated_in),
            'health_scores': json.dumps(result.health_scores),
        }
        df = pd.DataFrame([log_entry])
        if log_file.exists():
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, index=False)

        return result


def get_rotated_pairs(
    top_k: int = 3,
    force: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Convenience function: run rotation and return VALIDATED_PAIRS config dict.

    Use in run_paper_trading.py:
        from Paper_Trading.pair_rotation import get_rotated_pairs
        pairs = get_rotated_pairs(top_k=3)
    """
    manager = PairRotationManager(top_k=top_k)
    result = manager.run_rotation(force=force, verbose=verbose)

    if result is None and manager.current_pairs:
        # No rotation needed — use existing pairs
        scan = manager.scan_universe(verbose=False)
        return manager.build_validated_pairs_config(manager.current_pairs, scan)
    elif result is not None:
        scan = manager.scan_universe(verbose=False)
        return manager.build_validated_pairs_config(result.selected_pairs, scan)
    else:
        # Fallback to expanded universe top pairs
        scan = manager.scan_universe(verbose=verbose)
        rankings = manager.rank_pairs(scan)
        selected = manager.select_top_pairs(rankings, scan)
        if selected:
            return manager.build_validated_pairs_config(selected, scan)
        else:
            logger.error("No pairs available!")
            return {}


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pair Rotation Manager — scan, rank, and select healthiest pairs'
    )
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top pairs to select (default: 3)')
    parser.add_argument('--force', action='store_true',
                        help='Force rotation regardless of interval')
    parser.add_argument('--scan-only', action='store_true',
                        help='Only scan and rank, do not apply rotation')
    parser.add_argument('--interval', type=int, default=30,
                        help='Rotation interval in days (default: 30)')
    parser.add_argument('--min-health', type=float, default=0.50,
                        help='Minimum health score to be eligible (default: 0.50)')
    parser.add_argument('--min-healthy-pct', type=float, default=7.0,
                        help='Minimum recent healthy-day percent (default: 7.0)')
    parser.add_argument('--universe-mode', choices=['success', 'expanded', 'static'], default='success',
                        help='Universe source: success pairs, expanded universe, or static fallback')
    args = parser.parse_args()

    print(f"\nPair Rotation Manager")
    print(f"  Top-K: {args.top_k}")
    print(f"  Interval: {args.interval} days")
    print(f"  Min health: {args.min_health}")
    print(f"  Min healthy %: {args.min_healthy_pct}")
    print(f"  Universe mode: {args.universe_mode}")

    manager = PairRotationManager(
        top_k=args.top_k,
        rotation_interval_days=args.interval,
        min_health_score=args.min_health,
        min_healthy_pct=args.min_healthy_pct,
        universe_mode=args.universe_mode,
    )

    if args.scan_only:
        scan = manager.scan_universe(verbose=True)
        rankings = manager.rank_pairs(scan)
        print(f"\n  Rankings:")
        for i, (name, score, elig) in enumerate(rankings, 1):
            status = "eligible" if elig else "excluded"
            print(f"  {i}. {name:<15} score={score:.3f}  {status}")
    else:
        result = manager.run_rotation(force=args.force, verbose=True)
        if result:
            config = manager.build_validated_pairs_config(
                result.selected_pairs,
                manager.scan_universe(verbose=False),
            )
            print(f"\n  New VALIDATED_PAIRS config:")
            for pair, cfg in config.items():
                print(f"    {pair}: {cfg['ticker_y']}/{cfg['ticker_x']} "
                      f"(size={cfg['position_size']:.0%}, "
                      f"health={cfg.get('health_score', 0):.3f})")
