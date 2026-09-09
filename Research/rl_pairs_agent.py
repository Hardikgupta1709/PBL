import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import random

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, ConservativeKalman, download_data,
    calculate_zscore, calculate_half_life, generate_signals,
    StrictRegimeClassifier
)
from Research.cointegration_analysis import hurst_exponent
from Research.dynamic_pair_selector import (
    PairHealthMonitor, HealthThresholds
)

logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# 1. TRADING ENVIRONMENT (NO gymnasium DEPENDENCY)
# =============================================================================

@dataclass
class EnvConfig:
    """Environment configuration."""
    slippage_bps: float = 5.0
    commission_per_trade: float = 1.0
    notional_per_leg: float = 50000.0
    max_hold_days: int = 30
    stop_loss_z: float = 3.0          # Force exit if z exceeds this
    z_score_window: int = 40
    health_window: int = 126
    reward_scaling: float = 100.0      # Scale rewards for stable training


class PairsTradingEnv:
    """
    Custom trading environment for pairs trading.

    State vector (9 features):
        0: z_score (normalized)
        1: regime (0=crisis, 1=volatile, 2=normal → scaled to [0,1])
        2: spread_volatility (rolling 20d std of spread returns)
        3: hurst_estimate (trailing)
        4: half_life_estimate (trailing, normalized)
        5: position (-1, 0, 1 → scaled to [-1, 1])
        6: days_held (normalized by max_hold_days)
        7: rolling_pnl (trailing 20d PnL)
        8: health_score (pair health, 0-1)

    Actions:
        0: FLAT (close position / stay flat)
        1: LONG spread (buy Y, sell X)
        2: SHORT spread (sell Y, buy X)

    Reward:
        Step-wise realized + unrealized PnL minus transaction costs.
        No reward shaping to avoid bias.
    """

    N_STATES = 9
    N_ACTIONS = 3  # FLAT, LONG, SHORT

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.data = None
        self.current_step = 0
        self.position = 0         # -1, 0, 1
        self.entry_step = None
        self.entry_price_y = None
        self.entry_price_x = None
        self.entry_hedge = None
        self.cumulative_pnl = 0.0
        self.episode_trades = 0
        self.episode_pnls = []

    def setup(self, backtest_data: pd.DataFrame, health_data: pd.DataFrame = None):
        """
        Set up the environment with pre-computed backtest data.

        Parameters
        ----------
        backtest_data : pd.DataFrame
            Output from ConservativeSystem.run_backtest() containing:
            z_score, regime, spread, hedge_ratio, Y, X, etc.
        health_data : pd.DataFrame
            Output from PairHealthMonitor (optional).
        """
        self.data = backtest_data.copy().reset_index(drop=True)
        # Reindex health data to integer index matching bt_data
        if health_data is not None and len(health_data) > 0:
            # Map health from datetime index to integer index via bt_data's original index
            health_reindexed = health_data.reindex(
                backtest_data.index, method='ffill'
            ).fillna(0.5)
            health_reindexed.index = range(len(health_reindexed))
            self.health = health_reindexed
        else:
            self.health = health_data

        # Pre-compute features that the agent will observe
        self._precompute_features()

    def _precompute_features(self):
        """Pre-compute rolling features for the state vector."""
        d = self.data

        # Spread returns and volatility
        spread_ret = d['spread'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        self.spread_vol = spread_ret.rolling(20, min_periods=5).std().fillna(0)

        # Rolling Hurst (computed every 20 steps, ffilled)
        self.rolling_hurst_vals = pd.Series(0.5, index=d.index)
        window = min(self.config.health_window, 63)  # Shorter for speed
        for i in range(window, len(d), 20):
            try:
                h = hurst_exponent(d['spread'].iloc[i-window:i].values)
                self.rolling_hurst_vals.iloc[i] = h
            except Exception:
                pass
        self.rolling_hurst_vals = self.rolling_hurst_vals.replace(0, np.nan).ffill().fillna(0.5)

        # Rolling half-life
        self.rolling_hl = pd.Series(20.0, index=d.index)
        for i in range(window, len(d), 20):
            try:
                hl = calculate_half_life(d['spread'].iloc[i-window:i])
                self.rolling_hl.iloc[i] = hl
            except Exception:
                pass
        self.rolling_hl = self.rolling_hl.replace(0, np.nan).ffill().fillna(20.0)

        # Rolling PnL (using 20-day trailing sum)
        self.rolling_pnl = d['strategy_return'].rolling(20, min_periods=1).sum().fillna(0)

        # Health score
        if self.health is not None and 'health_score' in self.health.columns:
            self.health_scores = self.health['health_score'].reindex(
                d.index
            ).fillna(0.5)
        else:
            self.health_scores = pd.Series(0.5, index=d.index)

    def reset(self, start_idx: int = 0) -> np.ndarray:
        """Reset environment to start of episode."""
        self.current_step = start_idx
        self.position = 0
        self.entry_step = None
        self.entry_price_y = None
        self.entry_price_x = None
        self.entry_hedge = None
        self.cumulative_pnl = 0.0
        self.episode_trades = 0
        self.episode_pnls = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Build the state vector."""
        i = self.current_step
        d = self.data

        z = d['z_score'].iloc[i] if not pd.isna(d['z_score'].iloc[i]) else 0
        regime = d['regime'].iloc[i] / 2.0  # Scale to [0, 1]
        sprd_vol = self.spread_vol.iloc[i] * 100  # Scale up
        hurst = self.rolling_hurst_vals.iloc[i]
        hl = min(self.rolling_hl.iloc[i] / 60.0, 1.0)  # Normalize
        pos = float(self.position)
        days_held = 0
        if self.entry_step is not None:
            days_held = min(
                (self.current_step - self.entry_step) / self.config.max_hold_days,
                1.0
            )
        rpnl = self.rolling_pnl.iloc[i] * self.config.reward_scaling
        health = self.health_scores.iloc[i]

        state = np.array([
            np.clip(z / 4.0, -1, 1),     # Normalize z-score
            regime,
            np.clip(sprd_vol, 0, 1),
            np.clip(hurst, 0, 1),
            hl,
            pos,
            days_held,
            np.clip(rpnl, -1, 1),
            health,
        ], dtype=np.float32)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : int
            0=FLAT, 1=LONG, 2=SHORT

        Returns
        -------
        (next_state, reward, done, info)
        """
        i = self.current_step
        d = self.data

        old_position = self.position
        reward = 0.0
        info = {'trade': False, 'forced_exit': False}

        # Map action to target position
        target_position = {0: 0, 1: 1, 2: -1}[action]

        # CRISIS regime: block new entries (like the original strategy)
        regime = int(d['regime'].iloc[i])
        if regime == 0 and old_position == 0:
            target_position = 0  # Block entry during crisis

        # Force exit conditions
        z = d['z_score'].iloc[i] if not pd.isna(d['z_score'].iloc[i]) else 0
        if old_position != 0:
            days_held = i - self.entry_step if self.entry_step is not None else 0
            # Stop loss
            if abs(z) > self.config.stop_loss_z:
                target_position = 0
                info['forced_exit'] = True
            # Max hold
            elif days_held >= self.config.max_hold_days:
                target_position = 0
                info['forced_exit'] = True

        # Compute reward (PnL)
        if i > 0:
            spread_ret = d['spread_return'].iloc[i] if 'spread_return' in d.columns else 0
            # PnL from holding the old position
            step_pnl = old_position * spread_ret

            # Transaction cost if position changed
            position_change = abs(target_position - old_position)
            if position_change > 0:
                cost = (
                    position_change * (self.config.slippage_bps / 10_000) +
                    (self.config.commission_per_trade * 2) / self.config.notional_per_leg
                )
                step_pnl -= cost
                info['trade'] = True
                self.episode_trades += 1

            reward = step_pnl * self.config.reward_scaling
            self.cumulative_pnl += step_pnl
            self.episode_pnls.append(step_pnl)

        # Update position
        if target_position != old_position:
            if target_position != 0:
                self.entry_step = i
                self.entry_price_y = d['Y'].iloc[i]
                self.entry_price_x = d['X'].iloc[i]
                self.entry_hedge = d['hedge_ratio'].iloc[i]
            else:
                self.entry_step = None
                self.entry_price_y = None
                self.entry_price_x = None
                self.entry_hedge = None

        self.position = target_position

        # Advance
        self.current_step += 1
        done = self.current_step >= len(d) - 1

        next_state = self._get_state() if not done else np.zeros(self.N_STATES)

        return next_state, reward, done, info

    @property
    def n_steps(self):
        return len(self.data) if self.data is not None else 0


# =============================================================================
# 2. DQN NETWORK
# =============================================================================

class DQNetwork(nn.Module):
    """
    Deep Q-Network with dueling architecture.

    Small network to avoid overfitting:
        - 2 hidden layers × 64 units
        - Dropout for regularization
        - Dueling streams: value + advantage
    """

    def __init__(self, n_states: int = 9, n_actions: int = 3,
                 hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Dueling: separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Dueling combination
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# =============================================================================
# 3. EXPERIENCE REPLAY
# =============================================================================

class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 4. DQN AGENT
# =============================================================================

@dataclass
class AgentConfig:
    """DQN agent hyperparameters."""
    learning_rate: float = 1e-3
    gamma: float = 0.99            # Discount factor
    epsilon_start: float = 1.0     # Initial exploration
    epsilon_end: float = 0.05      # Minimum exploration
    epsilon_decay: float = 0.995   # Per-episode decay
    batch_size: int = 64
    buffer_size: int = 50000
    target_update_freq: int = 10   # Episodes between target network sync
    hidden_size: int = 64
    dropout: float = 0.1
    weight_decay: float = 1e-4     # L2 regularization
    max_episodes: int = 200
    early_stop_patience: int = 20  # Stop if val Sharpe doesn't improve


class DQNAgent:
    """
    DQN agent for pairs trading entry/exit decisions.

    Training follows strict temporal protocol:
        - Train on episodes sampled from training data
        - Validate on held-out period for early stopping
        - Test on unseen OOS data for final reporting
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.device = torch.device('cpu')  # CPU for reproducibility

        # Networks
        self.q_network = DQNetwork(
            n_states=PairsTradingEnv.N_STATES,
            n_actions=PairsTradingEnv.N_ACTIONS,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        ).to(self.device)

        self.target_network = DQNetwork(
            n_states=PairsTradingEnv.N_STATES,
            n_actions=PairsTradingEnv.N_ACTIONS,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.epsilon = self.config.epsilon_start
        self.training_history = []

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if explore and random.random() < self.epsilon:
            return random.randint(0, PairsTradingEnv.N_ACTIONS - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            q_values = self.q_network(state_t)
            self.q_network.train()
            return q_values.argmax(dim=1).item()

    def update(self) -> float:
        """One step of DQN training."""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Online network selects action
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            # Target network evaluates
            next_q = self.target_network(next_states_t)
            next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        # Huber loss (more robust than MSE)
        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """Hard update: copy online → target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_on_data(
        self,
        env: PairsTradingEnv,
        train_indices: range,
        val_indices: Optional[range] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the DQN agent on historical data.

        Each episode = one pass through the training data.
        Multiple episodes with different exploration noise provide diversity.

        Parameters
        ----------
        env : PairsTradingEnv
            Trading environment with data loaded.
        train_indices : range
            Indices of training data in env.data.
        val_indices : range, optional
            Indices for validation (early stopping).
        verbose : bool
            Print progress.

        Returns
        -------
        dict with training history.
        """
        best_val_sharpe = -np.inf
        patience_counter = 0
        best_weights = None

        for episode in range(self.config.max_episodes):
            # --- Training episode ---
            state = env.reset(start_idx=train_indices.start)
            total_reward = 0
            total_loss = 0
            n_updates = 0
            ep_returns = []

            for step in range(len(train_indices) - 1):
                if env.current_step >= train_indices.stop - 1:
                    break

                action = self.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Collect returns for Sharpe computation
                if len(env.episode_pnls) > 0:
                    ep_returns.append(env.episode_pnls[-1])

                # Train
                loss = self.update()
                if loss > 0:
                    total_loss += loss
                    n_updates += 1

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )

            # Sync target network
            if (episode + 1) % self.config.target_update_freq == 0:
                self.sync_target()

            # Compute training Sharpe
            ep_returns_arr = np.array(ep_returns)
            if len(ep_returns_arr) > 0 and ep_returns_arr.std() > 0:
                train_sharpe = (
                    ep_returns_arr.mean() / ep_returns_arr.std() * np.sqrt(252)
                )
            else:
                train_sharpe = 0.0

            avg_loss = total_loss / max(n_updates, 1)

            # --- Validation ---
            val_sharpe = np.nan
            if val_indices is not None and len(val_indices) > 0:
                val_sharpe = self._evaluate(env, val_indices)

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    patience_counter = 0
                    best_weights = {
                        k: v.clone() for k, v in self.q_network.state_dict().items()
                    }
                else:
                    patience_counter += 1

            self.training_history.append({
                'episode': episode + 1,
                'epsilon': self.epsilon,
                'avg_loss': avg_loss,
                'total_reward': total_reward,
                'train_sharpe': train_sharpe,
                'val_sharpe': val_sharpe,
                'n_trades': env.episode_trades,
            })

            if verbose and (episode + 1) % 10 == 0:
                val_str = f"  Val Sharpe: {val_sharpe:.3f}" if not np.isnan(val_sharpe) else ""
                print(f"  Episode {episode+1:3d} | "
                      f"ε={self.epsilon:.3f} | "
                      f"Loss={avg_loss:.4f} | "
                      f"Reward={total_reward:.2f} | "
                      f"Train Sharpe={train_sharpe:.3f} | "
                      f"Trades={env.episode_trades}{val_str}")

            # Early stopping
            if patience_counter >= self.config.early_stop_patience:
                if verbose:
                    print(f"\n  Early stopping at episode {episode+1} "
                          f"(val Sharpe not improving for {patience_counter} episodes)")
                break

        # Load best weights
        if best_weights is not None:
            self.q_network.load_state_dict(best_weights)
            self.sync_target()
            if verbose:
                print(f"  Loaded best weights (val Sharpe = {best_val_sharpe:.3f})")

        return {
            'history': self.training_history,
            'best_val_sharpe': best_val_sharpe,
            'total_episodes': len(self.training_history),
        }

    def _evaluate(self, env: PairsTradingEnv, indices: range) -> float:
        """Evaluate agent on a data slice (no exploration)."""
        state = env.reset(start_idx=indices.start)
        returns = []

        for step in range(len(indices) - 1):
            if env.current_step >= indices.stop - 1:
                break

            action = self.select_action(state, explore=False)
            next_state, reward, done, info = env.step(action)
            state = next_state

            if len(env.episode_pnls) > 0:
                returns.append(env.episode_pnls[-1])

            if done:
                break

        returns_arr = np.array(returns)
        if len(returns_arr) > 0 and returns_arr.std() > 0:
            sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        return sharpe

    def generate_signals(self, env: PairsTradingEnv, indices: range) -> pd.Series:
        """
        Generate trading signals using the trained agent.

        Returns pd.Series of signals {-1, 0, 1} aligned to the data index.
        """
        state = env.reset(start_idx=indices.start)
        signals = pd.Series(0, index=env.data.index)

        for step in range(len(indices) - 1):
            if env.current_step >= indices.stop - 1:
                break

            action = self.select_action(state, explore=False)
            target_pos = {0: 0, 1: 1, 2: -1}[action]

            # Record signal before stepping
            idx = env.current_step
            signals.iloc[idx] = target_pos

            next_state, reward, done, info = env.step(action)
            state = next_state

            if done:
                break

        return signals


# =============================================================================
# 5. FULL RL BACKTEST PIPELINE
# =============================================================================

def run_rl_backtest(
    ticker_y: str = 'BAC',
    ticker_x: str = 'PNC',
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    val_end_date: str = '2022-12-31',
    slippage_bps: float = 5.0,
    agent_config: Optional[AgentConfig] = None,
    env_config: Optional[EnvConfig] = None,
    verbose: bool = True,
) -> Dict:
    """
    Full RL backtest pipeline:
        1. Download data and run base backtest (for features)
        2. Set up trading environment
        3. Train DQN agent on training data with validation
        4. Generate RL signals on OOS data
        5. Compute returns and compare to fixed-threshold baseline

    Parameters
    ----------
    ticker_y, ticker_x : str
        Pair tickers.
    train_end_date : str
        End of training period.
    val_end_date : str
        End of validation period (for early stopping).
    slippage_bps : float
        Transaction costs.
    agent_config : AgentConfig
        DQN hyperparameters.
    env_config : EnvConfig
        Environment parameters.

    Returns
    -------
    dict with 'rl_metrics', 'baseline_metrics', 'comparison', 'agent',
    'training_history', 'signals_rl', 'signals_baseline'
    """
    if agent_config is None:
        agent_config = AgentConfig()
    if env_config is None:
        env_config = EnvConfig()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  RL PAIRS TRADING: {ticker_y}/{ticker_x}")
        print(f"  Train ≤ {train_end_date} | Val ≤ {val_end_date}")
        print(f"  Agent: DQN (hidden={agent_config.hidden_size}, "
              f"lr={agent_config.learning_rate}, γ={agent_config.gamma})")
        print(f"{'=' * 70}")

    # --- Step 1: Download data and run base backtest ---
    if verbose:
        print(f"\n  Step 1: Downloading data & running base backtest...")

    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    system = ConservativeSystem()
    bt_data = system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        slippage_bps=slippage_bps,
        verbose=False,
    )

    # Compute health metrics for state features
    if verbose:
        print(f"  Step 2: Computing pair health features...")

    try:
        monitor = PairHealthMonitor()
        health = monitor.compute_rolling_health(
            stock_y, stock_x, bt_data['spread'], step=5,
        )
    except Exception:
        health = None

    # --- Step 3: Set up environment ---
    if verbose:
        print(f"  Step 3: Setting up trading environment...")

    env = PairsTradingEnv(env_config)
    env.setup(bt_data, health)

    # Define temporal splits
    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)

    train_mask = bt_data.index <= train_end
    val_mask = (bt_data.index > train_end) & (bt_data.index <= val_end)
    test_mask = bt_data.index > val_end

    train_start_idx = 0
    train_end_idx = int(train_mask.sum())
    val_start_idx = train_end_idx
    val_end_idx = val_start_idx + int(val_mask.sum())
    test_start_idx = val_end_idx
    test_end_idx = len(bt_data)

    train_range = range(train_start_idx, train_end_idx)
    val_range = range(val_start_idx, val_end_idx) if val_mask.sum() > 0 else None
    test_range = range(test_start_idx, test_end_idx)

    if verbose:
        print(f"    Train: {len(train_range)} days | "
              f"Val: {len(val_range) if val_range else 0} days | "
              f"Test: {len(test_range)} days")

    # --- Step 4: Train DQN agent ---
    if verbose:
        print(f"\n  Step 4: Training DQN agent...")

    agent = DQNAgent(agent_config)
    training_result = agent.train_on_data(
        env, train_range, val_range, verbose=verbose,
    )

    # --- Step 5: Generate RL signals on full OOS period ---
    if verbose:
        print(f"\n  Step 5: Generating RL signals on OOS data...")

    oos_range = range(train_end_idx, test_end_idx)
    rl_signals = agent.generate_signals(env, oos_range)

    # --- Step 6: Compute RL returns ---
    if verbose:
        print(f"  Step 6: Computing RL returns...")

    oos_data = bt_data.iloc[train_end_idx:test_end_idx].copy()
    rl_sig_oos = rl_signals.iloc[train_end_idx:test_end_idx]

    # RL returns
    rl_returns = rl_sig_oos.shift(1) * oos_data['spread_return']
    rl_sig_changes = rl_sig_oos.diff().abs().fillna(0)
    rl_trades_occurred = (rl_sig_changes > 0).astype(float)
    rl_cost = (
        rl_sig_changes * (slippage_bps / 10_000) +
        rl_trades_occurred * (env_config.commission_per_trade * 2) / env_config.notional_per_leg
    )
    rl_net_returns = rl_returns - rl_cost
    rl_net_returns = rl_net_returns.dropna()

    # Baseline returns (already computed in bt_data)
    baseline_returns = oos_data['strategy_return'].dropna()

    # --- Step 7: Compute metrics ---
    def calc_metrics(returns, label):
        if len(returns) == 0 or returns.std() == 0:
            return {'sharpe': 0, 'total_return': 0, 'max_dd': 0,
                    'n_trades': 0, 'win_rate': 0, 'label': label}

        total = (1 + returns).prod() - 1
        n = len(returns)
        ann = (1 + total) ** (252 / n) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        cum = (1 + returns).cumprod()
        max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        win_rate = (returns > 0).mean()

        return {
            'sharpe': sharpe,
            'total_return': total,
            'annual_return': ann,
            'annual_vol': vol,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'n_days': n,
            'label': label,
        }

    rl_metrics = calc_metrics(rl_net_returns, 'RL Agent')

    # Count RL trades
    rl_n_trades = int((rl_sig_changes > 0).sum()) // 2
    rl_metrics['n_trades'] = rl_n_trades

    baseline_metrics = calc_metrics(baseline_returns, 'Fixed Threshold')
    baseline_metrics['n_trades'] = len(system.get_trade_analysis(period='oos'))

    # Comparison
    comparison = {
        'sharpe_delta': rl_metrics['sharpe'] - baseline_metrics['sharpe'],
        'return_delta': rl_metrics['total_return'] - baseline_metrics['total_return'],
        'dd_delta': rl_metrics['max_dd'] - baseline_metrics['max_dd'],
        'trade_delta': rl_metrics['n_trades'] - baseline_metrics['n_trades'],
        'rl_improved': rl_metrics['sharpe'] > baseline_metrics['sharpe'],
    }

    # --- Print results ---
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  RL AGENT vs FIXED THRESHOLD — OOS COMPARISON")
        print(f"{'=' * 70}")

        print(f"\n  {'Metric':<25} {'Fixed Threshold':>15} {'RL Agent':>15} {'Change':>12}")
        print(f"  {'─' * 67}")

        metrics_to_show = [
            ('OOS Sharpe', 'sharpe', '.3f'),
            ('OOS Return (%)', 'total_return', '%'),
            ('OOS Max DD (%)', 'max_dd', '%'),
            ('Trades', 'n_trades', 'd'),
            ('Win Rate', 'win_rate', '%'),
        ]

        for label, key, fmt in metrics_to_show:
            bl = baseline_metrics[key]
            rl = rl_metrics[key]
            if fmt == '%':
                print(f"  {label:<25} {bl*100:>14.2f}% {rl*100:>14.2f}% "
                      f"{(rl-bl)*100:>+11.2f}%")
            elif fmt == 'd':
                print(f"  {label:<25} {bl:>15d} {rl:>15d} {rl-bl:>+12d}")
            else:
                print(f"  {label:<25} {bl:>15{fmt}} {rl:>15{fmt}} "
                      f"{rl-bl:>+12{fmt}}")

        if comparison['rl_improved']:
            print(f"\n  ✅ RL Agent IMPROVED Sharpe by {comparison['sharpe_delta']:+.3f}")
        else:
            print(f"\n  ❌ RL Agent did NOT improve ({comparison['sharpe_delta']:+.3f})")

        print(f"\n  Training: {training_result['total_episodes']} episodes, "
              f"best val Sharpe = {training_result['best_val_sharpe']:.3f}")

    return {
        'rl_metrics': rl_metrics,
        'baseline_metrics': baseline_metrics,
        'comparison': comparison,
        'agent': agent,
        'training_history': training_result,
        'signals_rl': rl_signals,
        'signals_baseline': bt_data['final_signal'],
        'bt_data': bt_data,
        'rl_returns': rl_net_returns,
        'baseline_returns': baseline_returns,
    }


# =============================================================================
# 6. MULTI-PAIR RL ABLATION
# =============================================================================

def run_rl_ablation(
    pairs: List[Tuple[str, str]],
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    val_end_date: str = '2022-12-31',
    slippage_bps: float = 5.0,
    agent_config: Optional[AgentConfig] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run RL vs fixed threshold ablation across multiple pairs.
    """
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        )
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for i, (ty, tx) in enumerate(pairs):
        pair_name = f"{ty}_{tx}"
        print(f"\n{'━' * 70}")
        print(f"  [{i+1}/{len(pairs)}] {pair_name}")
        print(f"{'━' * 70}")

        try:
            result = run_rl_backtest(
                ty, tx,
                start_date=start_date,
                end_date=end_date,
                train_end_date=train_end_date,
                val_end_date=val_end_date,
                slippage_bps=slippage_bps,
                agent_config=agent_config,
                verbose=verbose,
            )

            rl = result['rl_metrics']
            bl = result['baseline_metrics']
            comp = result['comparison']

            rows.append({
                'Pair': pair_name,
                'Baseline_Sharpe': bl['sharpe'],
                'RL_Sharpe': rl['sharpe'],
                'Sharpe_Change': comp['sharpe_delta'],
                'Baseline_Return_%': bl['total_return'] * 100,
                'RL_Return_%': rl['total_return'] * 100,
                'Baseline_MaxDD_%': bl['max_dd'] * 100,
                'RL_MaxDD_%': rl['max_dd'] * 100,
                'Baseline_Trades': bl['n_trades'],
                'RL_Trades': rl['n_trades'],
                'RL_Improved': comp['rl_improved'],
                'Train_Episodes': result['training_history']['total_episodes'],
                'Best_Val_Sharpe': result['training_history']['best_val_sharpe'],
            })

        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        csv_path = os.path.join(save_dir, 'rl_ablation_summary.csv')
        summary.to_csv(csv_path, index=False)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  RL ABLATION SUMMARY ({len(summary)} pairs)")
            print(f"{'=' * 70}")
            print(f"\n  Pairs improved by RL: "
                  f"{summary['RL_Improved'].sum()}/{len(summary)}")
            print(f"  Avg Sharpe change: "
                  f"{summary['Sharpe_Change'].mean():+.3f}")

            print(f"\n  Per-pair:")
            for _, row in summary.iterrows():
                icon = "✅" if row['RL_Improved'] else "❌"
                print(f"    {icon} {row['Pair']:.<15} "
                      f"Baseline={row['Baseline_Sharpe']:.3f} → "
                      f"RL={row['RL_Sharpe']:.3f} "
                      f"({row['Sharpe_Change']:+.3f})")

            print(f"\n  Results saved to: {csv_path}")

    return summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 70)
    print("  WEEK 10: REINFORCEMENT LEARNING AGENT")
    print("=" * 70)

    # Test on existing 3 pairs
    test_pairs = [
        ('BAC', 'PNC'),
        ('WFC', 'MS'),
        ('CVX', 'OXY'),
    ]

    # Use modest config for faster training
    agent_cfg = AgentConfig(
        max_episodes=100,
        hidden_size=64,
        learning_rate=1e-3,
        epsilon_decay=0.99,
        early_stop_patience=15,
    )

    summary = run_rl_ablation(
        test_pairs,
        agent_config=agent_cfg,
    )

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
