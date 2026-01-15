# original 

"""
Streamlit Dashboard for Hybrid Pairs Trading System
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta

# Import your trading system classes
# Assuming your code is in a file called hybrid_pairs_trading.py
from hybrid_pairs_trading import (
    HybridPairsTradingSystem,
    download_data,
    KalmanFilterPairsTrading,
    MarketRegimeClassifier
)

# Page configuration
st.set_page_config(
    page_title="Hybrid Pairs Trading System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Hybrid Pairs Trading System</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Pair Selection
st.sidebar.subheader("Select Trading Pair")
pair_presets = {
    "PEP vs KO (Beverages)": ("PEP", "KO"),
    "JPM vs BAC (Banks)": ("JPM", "BAC"),
    "XOM vs CVX (Energy)": ("XOM", "CVX"),
    "MSFT vs ORCL (Tech)": ("MSFT", "ORCL"),
    "Custom": ("CUSTOM", "CUSTOM")
}

selected_pair = st.sidebar.selectbox(
    "Choose pair:",
    options=list(pair_presets.keys())
)

if selected_pair == "Custom":
    ticker_y = st.sidebar.text_input("Stock Y (Dependent):", "PEP")
    ticker_x = st.sidebar.text_input("Stock X (Independent):", "KO")
else:
    ticker_y, ticker_x = pair_presets[selected_pair]

# Date Range
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input(
    "Start Date",
    datetime(2018, 1, 1)
)
end_date = col2.date_input(
    "End Date",
    datetime(2024, 1, 1)
)

# Strategy Parameters
st.sidebar.subheader("Strategy Parameters")
entry_z = st.sidebar.slider(
    "Entry Z-Score Threshold",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1,
    help="Z-score threshold to enter positions"
)

exit_z = st.sidebar.slider(
    "Exit Z-Score Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Z-score threshold to exit positions"
)

train_period = st.sidebar.number_input(
    "Training Period (days)",
    min_value=100,
    max_value=500,
    value=252,
    step=50,
    help="Number of days to train ML model"
)

# Run Backtest Button
run_button = st.sidebar.button(" Run Backtest", type="primary", use_container_width=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'system' not in st.session_state:
    st.session_state.system = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Run Backtest
if run_button:
    with st.spinner(f"Running backtest for {ticker_y} vs {ticker_x}..."):
        try:
            # Download data
            progress_bar = st.progress(0)
            
            
            stock_y, stock_x, spy = download_data(
                ticker_y, 
                ticker_x, 
                'SPY',
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            progress_bar.progress(30)
            
            # Initialize system
           
            system = HybridPairsTradingSystem()
            progress_bar.progress(40)
            
            # Run backtest
            
            results = system.run_backtest(
                stock_y=stock_y,
                stock_x=stock_x,
                market_index=spy,
                train_period=train_period,
                entry_z=entry_z,
                exit_z=exit_z
            )
            progress_bar.progress(70)
            
            # Get metrics
            
            metrics = system.get_performance_metrics()
            progress_bar.progress(90)
            
            # Store in session state
            st.session_state.results = results
            st.session_state.system = system
            st.session_state.metrics = metrics
            st.session_state.ticker_y = ticker_y
            st.session_state.ticker_x = ticker_x
            
            progress_bar.progress(100)
     
            
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.stop()

# Display Results
if st.session_state.results is not None:
    results = st.session_state.results
    system = st.session_state.system
    metrics = st.session_state.metrics
    ticker_y = st.session_state.ticker_y
    ticker_x = st.session_state.ticker_x
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Overview", 
        " Performance", 
        " Signals", 
        " Risk Analysis",
        " Trade Log"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("Performance Overview")
        
        # Key Metrics Row 1
        col1, col2, col3, col4 = st.columns(4)
        
        hybrid_metrics = metrics['Hybrid Strategy']
        math_metrics = metrics['Math Only']
        
        with col1:
            st.metric(
                label=" Total Return ",
                value=hybrid_metrics['Total Return'],
                delta=f"vs Math Only: {math_metrics['Total Return']}"
            )
        
        with col2:
            st.metric(
                label=" Sharpe Ratio ",
                value=hybrid_metrics['Sharpe Ratio'],
                delta=f"vs Math Only: {math_metrics['Sharpe Ratio']}"
            )
        
        with col3:
            st.metric(
                label=" Max Drawdown (Hybrid)",
                value=hybrid_metrics['Max Drawdown'],
                delta=f"vs Math Only: {math_metrics['Max Drawdown']}",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label=" Win Rate (Hybrid)",
                value=hybrid_metrics['Win Rate'],
                delta=f"{hybrid_metrics['Total Trades']} trades"
            )
        
        st.markdown("---")
        
        # Cumulative Returns Chart
        st.subheader(" Cumulative Returns Comparison")
        
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Scatter(
            x=results.index,
            y=(results['strategy_cumulative'] - 1) * 100,
            name='Hybrid Strategy',
            line=dict(color='#2A9D8F', width=3),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=results.index,
            y=(results['math_only_cumulative'] - 1) * 100,
            name='Math Only',
            line=dict(color='#E76F51', width=2, dash='dash'),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
        fig_returns.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig_returns.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            legend=dict(x=0.02, y=0.98),
            template="plotly_white"
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hybrid Strategy Metrics")
            for key, value in hybrid_metrics.items():
                st.metric(label=key, value=value)
        
        with col2:
            st.subheader("Math Only Metrics")
            for key, value in math_metrics.items():
                st.metric(label=key, value=value)
    
    # TAB 2: PERFORMANCE
    with tab2:
        st.header("Detailed Performance Analysis")
        
        # Normalized Prices
        st.subheader(f"Stock Prices: {ticker_y} vs {ticker_x}")
        
        fig_prices = go.Figure()
        
        y_norm = (results['Y'] / results['Y'].iloc[0]) * 100
        x_norm = (results['X'] / results['X'].iloc[0]) * 100
        
        fig_prices.add_trace(go.Scatter(
            x=results.index,
            y=y_norm,
            name=ticker_y,
            line=dict(color='#2E86AB', width=2)
        ))
        
        fig_prices.add_trace(go.Scatter(
            x=results.index,
            y=x_norm,
            name=ticker_x,
            line=dict(color='#A23B72', width=2)
        ))
        
        # Shade unsafe regimes
        unsafe = results[results['regime'] == 0]
        if len(unsafe) > 0:
            for idx in unsafe.index:
                fig_prices.add_vrect(
                    x0=idx,
                    x1=idx + pd.Timedelta(days=1),
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0
                )
        
        fig_prices.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base=100)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_prices, use_container_width=True)
        
        # Drawdown Analysis
        st.subheader("Drawdown Analysis")
        
        hybrid_cum = results['strategy_cumulative']
        math_cum = results['math_only_cumulative']
        
        hybrid_dd = ((hybrid_cum - hybrid_cum.expanding().max()) / hybrid_cum.expanding().max()) * 100
        math_dd = ((math_cum - math_cum.expanding().max()) / math_cum.expanding().max()) * 100
        
        fig_dd = go.Figure()
        
        fig_dd.add_trace(go.Scatter(
            x=results.index,
            y=hybrid_dd,
            name='Hybrid',
            fill='tozeroy',
            line=dict(color='#2A9D8F', width=0),
            fillcolor='rgba(42, 157, 143, 0.5)'
        ))
        
        fig_dd.add_trace(go.Scatter(
            x=results.index,
            y=math_dd,
            name='Math Only',
            fill='tozeroy',
            line=dict(color='#E76F51', width=0),
            fillcolor='rgba(231, 111, 81, 0.3)'
        ))
        
        fig_dd.update_layout(
            height=350,
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Rolling Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rolling Sharpe Ratio (30-day)")
            rolling_sharpe = (
                results['strategy_return'].rolling(30).mean() / 
                results['strategy_return'].rolling(30).std() * np.sqrt(252)
            )
            
            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Scatter(
                x=results.index,
                y=rolling_sharpe,
                line=dict(color='#2E86AB', width=2),
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
            fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Good")
            fig_sharpe.add_hline(y=0.0, line_dash="dash", line_color="red", annotation_text="Breakeven")
            fig_sharpe.update_layout(height=300, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        with col2:
            st.subheader("Rolling Returns (30-day)")
            rolling_returns = results['strategy_return'].rolling(30).sum() * 100
            
            fig_roll_ret = go.Figure()
            fig_roll_ret.add_trace(go.Scatter(
                x=results.index,
                y=rolling_returns,
                line=dict(color='#A23B72', width=2),
                fill='tozeroy',
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
            fig_roll_ret.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_roll_ret.update_layout(height=300, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_roll_ret, use_container_width=True)
    
    # TAB 3: SIGNALS
    with tab3:
        st.header("Trading Signals Analysis")
        
        # Z-Score
        st.subheader("Z-Score & Trading Signals")
        
        fig_zscore = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=("Spread Z-Score", "Trading Signals"),
            vertical_spacing=0.1
        )
        
        # Z-Score plot
        z_data = results.dropna(subset=['z_score'])
        
        fig_zscore.add_trace(
            go.Scatter(
                x=z_data.index,
                y=z_data['z_score'],
                name='Z-Score',
                line=dict(color='#264653', width=2)
            ),
            row=1, col=1
        )
        
        # Threshold lines
        fig_zscore.add_hline(y=entry_z, line_dash="dash", line_color="red", row=1, col=1)
        fig_zscore.add_hline(y=-entry_z, line_dash="dash", line_color="red", row=1, col=1)
        fig_zscore.add_hline(y=exit_z, line_dash="dot", line_color="green", row=1, col=1)
        fig_zscore.add_hline(y=-exit_z, line_dash="dot", line_color="green", row=1, col=1)
        fig_zscore.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=1, col=1)
        
        # Signals plot
        fig_zscore.add_trace(
            go.Scatter(
                x=results.index,
                y=results['math_signal'],
                name='Math Signal',
                fill='tozeroy',
                line=dict(width=0),
                fillcolor='rgba(128, 128, 128, 0.3)'
            ),
            row=2, col=1
        )
        
        fig_zscore.add_trace(
            go.Scatter(
                x=results.index,
                y=results['final_signal'],
                name='Final Signal',
                fill='tozeroy',
                line=dict(width=0),
                fillcolor='rgba(42, 157, 143, 0.7)'
            ),
            row=2, col=1
        )
        
        fig_zscore.update_xaxes(title_text="Date", row=2, col=1)
        fig_zscore.update_yaxes(title_text="Z-Score", row=1, col=1)
        fig_zscore.update_yaxes(title_text="Position", row=2, col=1)
        
        fig_zscore.update_layout(
            height=600,
            hovermode='x unified',
            template="plotly_white"
        )
        
        st.plotly_chart(fig_zscore, use_container_width=True)
        
        # Signal Statistics
        col1, col2, col3 = st.columns(3)
        
        total_signals = (results['math_signal'] != 0).sum()
        filtered_signals = total_signals - (results['final_signal'] != 0).sum()
        executed_signals = (results['final_signal'] != 0).sum()
        
        with col1:
            st.metric("Math Signals Generated", total_signals)
        
        with col2:
            st.metric("Signals Filtered Out", filtered_signals, 
                     delta=f"{filtered_signals/total_signals*100:.1f}% filtered")
        
        with col3:
            st.metric("Signals Executed", executed_signals,
                     delta=f"{executed_signals/total_signals*100:.1f}% executed")
        
        # Hedge Ratio Evolution
        st.subheader(" Dynamic Hedge Ratio Evolution")
        
        fig_hedge = go.Figure()
        
        fig_hedge.add_trace(go.Scatter(
            x=results.index,
            y=results['hedge_ratio'],
            name='Hedge Ratio',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='β = %{y:.4f}<extra></extra>'
        ))
        
        fig_hedge.update_layout(
            height=300,
            xaxis_title="Date",
            yaxis_title="Hedge Ratio (β)",
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig_hedge, use_container_width=True)
        
        st.info(f"Current Hedge Ratio: {results['hedge_ratio'].iloc[-1]:.4f} (means: 1 share of {ticker_y} ≈ {results['hedge_ratio'].iloc[-1]:.4f} shares of {ticker_x})")
    
    # TAB 4: RISK ANALYSIS
    with tab4:
        st.header("Risk & Regime Analysis")
        
        # Regime Distribution
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Regime Distribution")
            
            safe_days = (results['regime'] == 1).sum()
            unsafe_days = (results['regime'] == 0).sum()
            total_days = len(results)
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['SAFE', 'UNSAFE'],
                values=[safe_days, unsafe_days],
                marker_colors=['#2A9D8F', '#E76F51'],
                hole=0.4,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig_pie.update_layout(
                height=300,
                showlegend=True,
                annotations=[dict(text=f'{total_days}<br>days', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.metric("Total Trading Days", total_days)
            st.metric("Safe Regime Days", safe_days, 
                     delta=f"{safe_days/total_days*100:.1f}%")
            st.metric("Unsafe Regime Days", unsafe_days,
                     delta=f"{unsafe_days/total_days*100:.1f}%",
                     delta_color="inverse")
        
        with col2:
            st.subheader(" Market Volatility Over Time")
            
            vol = results['Market'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=results.index,
                y=vol,
                name='20-day Volatility',
                line=dict(color='#264653', width=2),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
            
            # Shade unsafe regions
            unsafe = results[results['regime'] == 0]
            if len(unsafe) > 0:
                for idx in unsafe.index:
                    fig_vol.add_vrect(
                        x0=idx,
                        x1=idx + pd.Timedelta(days=1),
                        fillcolor="red",
                        opacity=0.15,
                        layer="below",
                        line_width=0
                    )
            
            fig_vol.add_hline(y=25, line_dash="dash", line_color="red", 
                             annotation_text="Unsafe Threshold (25%)")
            
            fig_vol.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="Annualized Volatility (%)",
                template="plotly_white",
                showlegend=False
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        
        st.markdown("---")
        
        # Returns by Regime
        st.subheader(" Performance Comparison by Regime")
        
        safe_ret = results[results['regime'] == 1]['strategy_return'].dropna() * 100
        unsafe_ret_if_traded = results[results['regime'] == 0]['math_signal'].shift(1) * results['spread_return'] * 100
        unsafe_ret_if_traded = unsafe_ret_if_traded[unsafe_ret_if_traded != 0].dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "SAFE Regime Avg Return",
                f"{safe_ret.mean():.3f}%",
                delta=f"σ = {safe_ret.std():.3f}%"
            )
        
        with col2:
            st.metric(
                "UNSAFE (if traded) Avg Return",
                f"{unsafe_ret_if_traded.mean():.3f}%",
                delta=f"σ = {unsafe_ret_if_traded.std():.3f}%",
                delta_color="inverse"
            )
        
        with col3:
            avoided_loss = (unsafe_ret_if_traded.mean() * len(unsafe_ret_if_traded))
            st.metric(
                "Loss Avoided by ML Filter",
                f"{avoided_loss:.2f}%",
                delta="Total cumulative"
            )
        
        # Box plot
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=safe_ret,
            name='SAFE (Traded)',
            marker_color='#2A9D8F',
            boxmean='sd'
        ))
        
        fig_box.add_trace(go.Box(
            y=unsafe_ret_if_traded,
            name='UNSAFE (Avoided)',
            marker_color='#E76F51',
            boxmean='sd'
        ))
        
        fig_box.update_layout(
            height=400,
            yaxis_title="Daily Return (%)",
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
       
    
    # TAB 5: TRADE LOG
    with tab5:
        st.header("Trade History & Analysis")
        
        trades = system.get_trade_analysis()
        
        if len(trades) > 0:
            # Summary Stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", len(trades))
            
            with col2:
                winning_trades = trades['profitable'].sum()
                st.metric("Winning Trades", winning_trades,
                         delta=f"{winning_trades/len(trades)*100:.1f}% win rate")
            
            with col3:
                st.metric("Average Duration", f"{trades['duration_days'].mean():.1f} days")
            
            with col4:
                avg_pnl = trades['pnl_pct'].mean()
                st.metric("Average P&L", f"{avg_pnl:.2f}%",
                         delta="per trade")
            
            st.markdown("---")
            
            # Trade Performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Trade P&L Distribution")
                
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=trades['pnl_pct'],
                    nbinsx=20,
                    marker_color='#2E86AB',
                    opacity=0.7,
                    name='P&L Distribution'
                ))
                
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red",
                                  annotation_text="Breakeven")
                
                fig_hist.update_layout(
                    height=300,
                    xaxis_title="P&L (%)",
                    yaxis_title="Number of Trades",
                    template="plotly_white",
                    showlegend=False
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader(" Trade Duration vs P&L")
                
                fig_scatter = go.Figure()
                
                colors = ['green' if p else 'red' for p in trades['profitable']]
                
                fig_scatter.add_trace(go.Scatter(
                    x=trades['duration_days'],
                    y=trades['pnl_pct'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors,
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Entry: {row['entry_date'].strftime('%Y-%m-%d')}<br>Exit: {row['exit_date'].strftime('%Y-%m-%d')}<br>P&L: {row['pnl_pct']:.2f}%" 
                          for _, row in trades.iterrows()],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig_scatter.update_layout(
                    height=300,
                    xaxis_title="Duration (days)",
                    yaxis_title="P&L (%)",
                    template="plotly_white",
                    showlegend=False
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed Trade Table
            st.subheader("Detailed Trade Log")
            
            # Format trades for display
            display_trades = trades.copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            display_trades['entry_z'] = display_trades['entry_z'].round(2)
            display_trades['exit_z'] = display_trades['exit_z'].round(2)
            display_trades['entry_hedge'] = display_trades['entry_hedge'].round(4)
            display_trades['pnl_pct'] = display_trades['pnl_pct'].round(2)
            display_trades['profitable'] = display_trades['profitable'].map({True: '1', False: '0'})
            
            # Rename columns
            display_trades = display_trades.rename(columns={
                'entry_date': 'Entry Date',
                'exit_date': 'Exit Date',
                'entry_signal': 'Signal',
                'entry_z': 'Entry Z',
                'exit_z': 'Exit Z',
                'entry_hedge': 'Hedge Ratio',
                'duration_days': 'Days',
                'pnl_pct': 'P&L (%)',
                'profitable': 'Result'
            })
            
            # Display with styling
            st.dataframe(
                display_trades[['Entry Date', 'Exit Date', 'Signal', 'Entry Z', 'Exit Z', 
                               'Hedge Ratio', 'Days', 'P&L (%)', 'Result']],
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = trades.to_csv(index=False)
            st.download_button(
                label="Download Trade Log CSV",
                data=csv,
                file_name=f'trades_{ticker_y}_{ticker_x}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
            
        else:
            st.warning("No complete trades found in the backtest period.")

else:
    # Welcome screen when no backtest has been run

    
    st.markdown("---")
    
    # System Overview
    st.header("Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  Mathematical Component
        **Kalman Filter for Pairs Trading**
        
        - Dynamically estimates hedge ratio between stocks
        - Identifies mean-reversion opportunities
        - Generates precise entry/exit signals
        - Based on state-space modeling
        
        **Key Features:**
        - Adaptive to changing relationships
        - Optimal estimation under Gaussian noise
        - Real-time hedge ratio updates
        """)
    
    with col2:
        st.markdown("""
        ###  Machine Learning Component
        **Random Forest Regime Classifier**
        
        - Detects unsafe market conditions
        - Filters trading signals during crises
        - Uses volatility, drawdown, correlation features
        - Acts as risk management layer
        
        **Key Features:**
        - Pattern recognition for market stress
        - Binary classification (SAFE/UNSAFE)
        - Explainable decisions
        """)
    
    st.markdown("---")
    
    # How it Works
    st.header("How the Hybrid System Works")
    
    st.markdown("""
    ### Decision Flow:
    
    1. **Kalman Filter** analyzes price data and generates trading signal
       - `z-score > 2.0` → **Short spread** (sell Y, buy X)
       - `z-score < -2.0` → **Long spread** (buy Y, sell X)
       - `|z-score| < 0.5` → **Exit position**
    
    2. **ML Classifier** evaluates current market regime
       - Analyzes 8 market features
       - Outputs: `SAFE (1)` or `UNSAFE (0)`
    
    3. **AND Logic Integration**
       - `Final Signal = Math Signal × Regime`
       - Trade executes **only if both agree**
    
    4. **Result:** Higher Sharpe ratio, lower drawdown
    """)
    

    
    # # Quick Start Guide
    # with st.expander("📚 Quick Start Guide"):
    #     st.markdown("""
    #     ### Getting Started:
        
    #     1. **Select a Trading Pair**
    #        - Choose from presets or enter custom tickers
    #        - Popular pairs: PEP/KO, JPM/BAC, XOM/CVX
        
    #     2. **Set Date Range**
    #        - Recommended: At least 3 years of data
    #        - Include at least one market crisis period
        
    #     3. **Configure Parameters**
    #        - Entry Z-Score: Typically 2.0 (2 std dev)
    #        - Exit Z-Score: Typically 0.5
    #        - Training Period: 252 days (1 year)
        
    #     4. **Run Backtest**
    #        - Click the "Run Backtest" button
    #        - Wait for data download and processing
    #        - Explore results in tabs
        
    #     5. **Analyze Results**
    #        - Overview: Key metrics comparison
    #        - Performance: Charts and rolling metrics
    #        - Signals: Z-scores and trading signals
    #        - Risk: Regime analysis
    #        - Trades: Detailed trade log
    #     """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Hybrid Pairs Trading System</strong> | Hardik Gupta</p>

</div>
""", unsafe_allow_html=True)




# # validation 

# """
# Streamlit Dashboard for Hybrid Pairs Trading System
# Complete interactive web application for strategy analysis and validation
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.express as px
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # Import your trading system and validator
# try:
#     from hybrid_pairs_trading import (
#         HybridPairsTradingSystem, 
#         download_data,
#         TradingVisualizer
#     )
#     from strategy_validator import StrategyValidator, run_complete_validation
#     IMPORTS_SUCCESS = True
# except ImportError as e:
#     IMPORTS_SUCCESS = False
#     IMPORT_ERROR = str(e)

# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================

# st.set_page_config(
#     page_title="Hybrid Pairs Trading System",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         color: #2E86AB;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #A23B72;
#         margin-top: 2rem;
#         margin-bottom: 1rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #2E86AB;
#     }
#     .success-box {
#         background-color: #d4edda;
#         border: 1px solid #c3e6cb;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         color: #155724;
#     }
#     .warning-box {
#         background-color: #fff3cd;
#         border: 1px solid #ffeaa7;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         color: #856404;
#     }
#     .danger-box {
#         background-color: #f8d7da;
#         border: 1px solid #f5c6cb;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         color: #721c24;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ============================================================================
# # SIDEBAR CONFIGURATION
# # ============================================================================

# def render_sidebar():
#     """Render sidebar with parameters and controls."""
#     st.sidebar.image("https://img.icons8.com/color/96/000000/stock-market.png", width=100)
#     st.sidebar.title("⚙️ Configuration")
    
#     # Navigation
#     page = st.sidebar.radio(
#         "Navigation",
#         ["🏠 Home", "📊 Backtest", "✅ Validation", "📈 Live Analysis", "📚 Documentation"]
#     )
    
#     st.sidebar.markdown("---")
    
#     # Stock pair selection
#     st.sidebar.subheader("Stock Pair Selection")
    
#     pair_preset = st.sidebar.selectbox(
#         "Select Preset Pair",
#         ["Custom", "PEP vs KO", "GLD vs GDX", "XLE vs XOM", "JPM vs BAC"]
#     )
    
#     if pair_preset == "Custom":
#         ticker_y = st.sidebar.text_input("Stock Y (Dependent)", "PEP")
#         ticker_x = st.sidebar.text_input("Stock X (Independent)", "KO")
#     else:
#         pairs_map = {
#             "PEP vs KO": ("PEP", "KO"),
#             "GLD vs GDX": ("GLD", "GDX"),
#             "XLE vs XOM": ("XLE", "XOM"),
#             "JPM vs BAC": ("JPM", "BAC")
#         }
#         ticker_y, ticker_x = pairs_map[pair_preset]
    
#     market_ticker = st.sidebar.text_input("Market Index", "SPY")
    
#     st.sidebar.markdown("---")
    
#     # Date range
#     st.sidebar.subheader("Date Range")
#     col1, col2 = st.sidebar.columns(2)
#     with col1:
#         start_date = st.date_input(
#             "Start Date",
#             value=datetime(2018, 1, 1),
#             max_value=datetime.now()
#         )
#     with col2:
#         end_date = st.date_input(
#             "End Date",
#             value=datetime(2024, 1, 1),
#             max_value=datetime.now()
#         )
    
#     st.sidebar.markdown("---")
    
#     # Strategy parameters
#     st.sidebar.subheader("Strategy Parameters")
    
#     entry_z = st.sidebar.slider(
#         "Entry Z-Score Threshold",
#         min_value=1.0,
#         max_value=3.0,
#         value=2.0,
#         step=0.1,
#         help="Z-score threshold to enter position"
#     )
    
#     exit_z = st.sidebar.slider(
#         "Exit Z-Score Threshold",
#         min_value=0.0,
#         max_value=1.5,
#         value=0.5,
#         step=0.1,
#         help="Z-score threshold to exit position"
#     )
    
#     train_period = st.sidebar.slider(
#         "Training Period (days)",
#         min_value=126,
#         max_value=504,
#         value=252,
#         step=63,
#         help="Number of days to train regime classifier"
#     )
    
#     st.sidebar.markdown("---")
    
#     # Validation parameters
#     st.sidebar.subheader("Validation Parameters")
    
#     n_folds = st.sidebar.slider(
#         "Walk-Forward Folds",
#         min_value=3,
#         max_value=10,
#         value=4,
#         help="Number of out-of-sample test periods"
#     )
    
#     n_simulations = st.sidebar.slider(
#         "Monte Carlo Simulations",
#         min_value=100,
#         max_value=5000,
#         value=1000,
#         step=100,
#         help="Number of simulations for statistical testing"
#     )
    
#     return {
#         'page': page,
#         'ticker_y': ticker_y,
#         'ticker_x': ticker_x,
#         'market_ticker': market_ticker,
#         'start_date': start_date.strftime('%Y-%m-%d'),
#         'end_date': end_date.strftime('%Y-%m-%d'),
#         'entry_z': entry_z,
#         'exit_z': exit_z,
#         'train_period': train_period,
#         'n_folds': n_folds,
#         'n_simulations': n_simulations
#     }

# # ============================================================================
# # HOME PAGE
# # ============================================================================

# def render_home_page():
#     """Render home page with system overview."""
#     st.markdown('<div class="main-header">🚀 Hybrid Pairs Trading System</div>', unsafe_allow_html=True)
    
#     st.markdown("""
#     ### Welcome to the Advanced Pairs Trading Strategy Dashboard
    
#     This system combines **mathematical precision** with **machine learning intelligence** to trade 
#     cointegrated stock pairs while avoiding dangerous market conditions.
#     """)
    
#     # Key features
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         #### 🧮 Mathematical Core
#         - Kalman Filter for dynamic hedge ratio
#         - Z-score based mean reversion
#         - Statistical arbitrage signals
#         """)
    
#     with col2:
#         st.markdown("""
#         #### 🤖 Machine Learning
#         - Random Forest regime classifier
#         - Crisis period detection
#         - Adaptive risk management
#         """)
    
#     with col3:
#         st.markdown("""
#         #### ✅ Robust Validation
#         - Walk-forward analysis
#         - Monte Carlo simulation
#         - Stress testing
#         """)
    
#     st.markdown("---")
    
#     # System architecture
#     st.markdown('<div class="sub-header">📐 System Architecture</div>', unsafe_allow_html=True)
    
#     st.info("""
#     **How It Works:**
    
#     1. **Kalman Filter** tracks the relationship between two cointegrated stocks
#     2. **Z-Score Analysis** identifies when the spread is extreme (trading opportunity)
#     3. **ML Regime Classifier** determines if market conditions are safe
#     4. **Hybrid Logic**: Trade only when `(Math Signal) AND (Safe Regime)`
#     5. **Result**: Higher returns with lower risk during crises
#     """)
    
#     # Quick start
#     st.markdown('<div class="sub-header">🚀 Quick Start Guide</div>', unsafe_allow_html=True)
    
#     st.markdown("""
#     1. **📊 Backtest Tab**: Run historical simulations and analyze performance
#     2. **✅ Validation Tab**: Verify strategy robustness with industry-standard tests
#     3. **📈 Live Analysis Tab**: Monitor real-time signals and positions
#     4. **📚 Documentation Tab**: Learn about the methodology and implementation
    
#     ⬅️ **Configure your strategy parameters in the sidebar to begin!**
#     """)
    
#     # Warning
#     st.warning("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice. Always backtest thoroughly before live trading.")

# # ============================================================================
# # BACKTEST PAGE
# # ============================================================================

# def render_backtest_page(config):
#     """Render backtest analysis page."""
#     st.markdown('<div class="main-header">📊 Strategy Backtest</div>', unsafe_allow_html=True)
    
#     st.markdown(f"""
#     **Current Configuration:**
#     - Pair: {config['ticker_y']} vs {config['ticker_x']}
#     - Period: {config['start_date']} to {config['end_date']}
#     - Entry Z-Score: {config['entry_z']}, Exit Z-Score: {config['exit_z']}
#     """)
    
#     # Run backtest button
#     if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
#         with st.spinner("📥 Downloading data..."):
#             try:
#                 stock_y, stock_x, market = download_data(
#                     config['ticker_y'],
#                     config['ticker_x'],
#                     config['market_ticker'],
#                     config['start_date'],
#                     config['end_date']
#                 )
#                 st.success(f"✅ Downloaded {len(stock_y)} days of data")
#             except Exception as e:
#                 st.error(f"❌ Error downloading data: {e}")
#                 return
        
#         with st.spinner("🔄 Running backtest..."):
#             try:
#                 system = HybridPairsTradingSystem()
#                 results = system.run_backtest(
#                     stock_y, stock_x, market,
#                     train_period=config['train_period'],
#                     entry_z=config['entry_z'],
#                     exit_z=config['exit_z']
#                 )
                
#                 # Store in session state
#                 st.session_state['system'] = system
#                 st.session_state['results'] = results
#                 st.session_state['stock_y'] = stock_y
#                 st.session_state['stock_x'] = stock_x
#                 st.session_state['market'] = market
#                 st.session_state['config'] = config
                
#                 st.success("✅ Backtest completed successfully!")
#             except Exception as e:
#                 st.error(f"❌ Error running backtest: {e}")
#                 return
    
#     # Display results if available
#     if 'results' in st.session_state:
#         display_backtest_results(
#             st.session_state['system'],
#             st.session_state['results'],
#             config
#         )

# def display_backtest_results(system, results, config):
#     """Display comprehensive backtest results."""
    
#     # Performance metrics
#     st.markdown('<div class="sub-header">📈 Performance Metrics</div>', unsafe_allow_html=True)
    
#     metrics = system.get_performance_metrics()
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("#### 🎯 Hybrid Strategy")
#         hybrid_metrics = metrics['Hybrid Strategy']
        
#         metric_cols = st.columns(3)
#         metric_cols[0].metric("Total Return", hybrid_metrics['Total Return'])
#         metric_cols[1].metric("Sharpe Ratio", hybrid_metrics['Sharpe Ratio'])
#         metric_cols[2].metric("Max Drawdown", hybrid_metrics['Max Drawdown'])
        
#         metric_cols = st.columns(3)
#         metric_cols[0].metric("Annual Return", hybrid_metrics['Annual Return'])
#         metric_cols[1].metric("Win Rate", hybrid_metrics['Win Rate'])
#         metric_cols[2].metric("Total Trades", hybrid_metrics['Total Trades'])
    
#     with col2:
#         st.markdown("#### 🧮 Math Only Strategy")
#         math_metrics = metrics['Math Only']
        
#         metric_cols = st.columns(3)
#         metric_cols[0].metric("Total Return", math_metrics['Total Return'])
#         metric_cols[1].metric("Sharpe Ratio", math_metrics['Sharpe Ratio'])
#         metric_cols[2].metric("Max Drawdown", math_metrics['Max Drawdown'])
        
#         metric_cols = st.columns(3)
#         metric_cols[0].metric("Annual Return", math_metrics['Annual Return'])
#         metric_cols[1].metric("Win Rate", math_metrics['Win Rate'])
#         metric_cols[2].metric("Total Trades", math_metrics['Total Trades'])
    
#     # Additional info
#     st.markdown("#### ℹ️ Additional Information")
#     info_cols = st.columns(4)
#     info = metrics['Additional Info']
#     info_cols[0].metric("Trading Days", info['Total Trading Days'])
#     info_cols[1].metric("Safe Regime Days", info['Safe Regime Days'])
#     info_cols[2].metric("Unsafe Regime Days", info['Unsafe Regime Days'])
#     info_cols[3].metric("Filter Rate", info['Regime Filter Rate'])
    
#     st.markdown("---")
    
#     # Visualizations
#     st.markdown('<div class="sub-header">📊 Interactive Visualizations</div>', unsafe_allow_html=True)
    
#     # Create tabs for different visualizations
#     tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📉 Z-Score", "🎯 Signals", "💰 Trades"])
    
#     with tab1:
#         fig = create_performance_chart(results, config)
#         st.plotly_chart(fig, use_container_width=True)
    
#     with tab2:
#         fig = create_zscore_chart(results, config)
#         st.plotly_chart(fig, use_container_width=True)
    
#     with tab3:
#         fig = create_signals_chart(results, config)
#         st.plotly_chart(fig, use_container_width=True)
    
#     with tab4:
#         trades = system.get_trade_analysis()
#         if len(trades) > 0:
#             display_trade_analysis(trades)
#         else:
#             st.info("No complete trades in this period")
    
#     st.markdown("---")
    
#     # Regime analysis
#     st.markdown('<div class="sub-header">🌐 Market Regime Analysis</div>', unsafe_allow_html=True)
#     fig = create_regime_chart(results)
#     st.plotly_chart(fig, use_container_width=True)

# def create_performance_chart(results, config):
#     """Create interactive performance comparison chart."""
#     fig = go.Figure()
    
#     fig.add_trace(go.Scatter(
#         x=results.index,
#         y=(results['strategy_cumulative'] - 1) * 100,
#         name='Hybrid Strategy',
#         line=dict(color='#2A9D8F', width=3),
#         hovertemplate='<b>Hybrid Strategy</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
#     ))
    
#     fig.add_trace(go.Scatter(
#         x=results.index,
#         y=(results['math_only_cumulative'] - 1) * 100,
#         name='Math Only',
#         line=dict(color='#E76F51', width=2, dash='dash'),
#         hovertemplate='<b>Math Only</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
#     ))
    
#     fig.update_layout(
#         title=f"Cumulative Returns: {config['ticker_y']} vs {config['ticker_x']}",
#         xaxis_title="Date",
#         yaxis_title="Cumulative Return (%)",
#         hovermode='x unified',
#         height=500,
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         )
#     )
    
#     return fig

# def create_zscore_chart(results, config):
#     """Create Z-score visualization with entry/exit zones."""
#     fig = go.Figure()
    
#     # Z-score line
#     fig.add_trace(go.Scatter(
#         x=results.index,
#         y=results['z_score'],
#         name='Z-Score',
#         line=dict(color='#264653', width=2),
#         hovertemplate='<b>Z-Score</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
#     ))
    
#     # Entry thresholds
#     fig.add_hline(y=config['entry_z'], line_dash="dash", line_color="red", 
#                   annotation_text=f"Entry (+{config['entry_z']}σ)")
#     fig.add_hline(y=-config['entry_z'], line_dash="dash", line_color="red",
#                   annotation_text=f"Entry (-{config['entry_z']}σ)")
    
#     # Exit thresholds
#     fig.add_hline(y=config['exit_z'], line_dash="dot", line_color="green",
#                   annotation_text=f"Exit (+{config['exit_z']}σ)")
#     fig.add_hline(y=-config['exit_z'], line_dash="dot", line_color="green",
#                   annotation_text=f"Exit (-{config['exit_z']}σ)")
    
#     # Neutral line
#     fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.5)
    
#     # Shaded regions
#     fig.add_hrect(y0=config['entry_z'], y1=4, fillcolor="red", opacity=0.1, line_width=0)
#     fig.add_hrect(y0=-config['entry_z'], y1=-4, fillcolor="red", opacity=0.1, line_width=0)
    
#     fig.update_layout(
#         title="Spread Z-Score with Trading Thresholds",
#         xaxis_title="Date",
#         yaxis_title="Z-Score",
#         hovermode='x unified',
#         height=500,
#         yaxis=dict(range=[-4, 4])
#     )
    
#     return fig

# def create_signals_chart(results, config):
#     """Create trading signals visualization."""
#     fig = go.Figure()
    
#     # Math signal
#     fig.add_trace(go.Scatter(
#         x=results.index,
#         y=results['math_signal'],
#         name='Math Signal',
#         fill='tozeroy',
#         fillcolor='rgba(128,128,128,0.3)',
#         line=dict(color='gray', width=0),
#         hovertemplate='<b>Math Signal</b><br>Date: %{x}<br>Position: %{y}<extra></extra>'
#     ))
    
#     # Final signal
#     fig.add_trace(go.Scatter(
#         x=results.index,
#         y=results['final_signal'],
#         name='Final Signal (Hybrid)',
#         fill='tozeroy',
#         fillcolor='rgba(42,157,143,0.7)',
#         line=dict(color='#2A9D8F', width=2),
#         hovertemplate='<b>Final Signal</b><br>Date: %{x}<br>Position: %{y}<extra></extra>'
#     ))
    
#     fig.update_layout(
#         title="Trading Signals: Math vs Hybrid",
#         xaxis_title="Date",
#         yaxis_title="Position",
#         hovermode='x unified',
#         height=500,
#         yaxis=dict(
#             tickmode='array',
#             tickvals=[-1, 0, 1],
#             ticktext=['Short', 'Flat', 'Long']
#         )
#     )
    
#     return fig

# def create_regime_chart(results):
#     """Create regime analysis visualization."""
#     fig = make_subplots(
#         rows=2, cols=1,
#         subplot_titles=("Market Volatility & Regime", "Regime Distribution"),
#         row_heights=[0.7, 0.3],
#         vertical_spacing=0.15
#     )
    
#     # Volatility
#     vol = results['Market'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    
#     fig.add_trace(go.Scatter(
#         x=results.index,
#         y=vol,
#         name='Market Volatility',
#         line=dict(color='#264653', width=2)
#     ), row=1, col=1)
    
#     # Regime shading
#     unsafe_periods = results[results['regime'] == 0]
#     if len(unsafe_periods) > 0:
#         for idx in unsafe_periods.index:
#             fig.add_vrect(
#                 x0=idx, x1=idx + timedelta(days=1),
#                 fillcolor="red", opacity=0.2,
#                 layer="below", line_width=0,
#                 row=1, col=1
#             )
    
#     # Regime distribution
#     regime_counts = results['regime'].value_counts()
    
#     fig.add_trace(go.Bar(
#         x=['UNSAFE', 'SAFE'],
#         y=[regime_counts.get(0, 0), regime_counts.get(1, 0)],
#         marker_color=['#E76F51', '#2A9D8F'],
#         text=[regime_counts.get(0, 0), regime_counts.get(1, 0)],
#         textposition='auto',
#         showlegend=False
#     ), row=2, col=1)
    
#     fig.update_xaxes(title_text="Date", row=1, col=1)
#     fig.update_xaxes(title_text="Regime Type", row=2, col=1)
#     fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
#     fig.update_yaxes(title_text="Number of Days", row=2, col=1)
    
#     fig.update_layout(height=700, showlegend=False, hovermode='x unified')
    
#     return fig

# def display_trade_analysis(trades):
#     """Display detailed trade analysis."""
#     st.markdown("#### 📊 Trade Statistics")
    
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Total Trades", len(trades))
#     col2.metric("Winning Trades", trades['profitable'].sum())
#     col3.metric("Losing Trades", (~trades['profitable']).sum())
#     col4.metric("Win Rate", f"{trades['profitable'].mean()*100:.1f}%")
    
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Avg Duration", f"{trades['duration_days'].mean():.1f} days")
#     col2.metric("Avg Win", f"{trades[trades['profitable']]['pnl_pct'].mean():.2f}%")
#     col3.metric("Avg Loss", f"{trades[~trades['profitable']]['pnl_pct'].mean():.2f}%")
#     col4.metric("Best Trade", f"{trades['pnl_pct'].max():.2f}%")
    
#     st.markdown("#### 📋 Recent Trades")
    
#     # Display table
#     display_trades = trades[['entry_date', 'exit_date', 'duration_days', 'pnl_pct', 'profitable']].copy()
#     display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
#     display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
#     display_trades['pnl_pct'] = display_trades['pnl_pct'].apply(lambda x: f"{x:.2f}%")
#     display_trades['profitable'] = display_trades['profitable'].apply(lambda x: "✅" if x else "❌")
    
#     st.dataframe(
#         display_trades.tail(15),
#         use_container_width=True,
#         hide_index=True
#     )
    
#     # Trade distribution
#     st.markdown("#### 📊 P&L Distribution")
#     fig = px.histogram(
#         trades,
#         x='pnl_pct',
#         nbins=20,
#         color='profitable',
#         color_discrete_map={True: '#2A9D8F', False: '#E76F51'},
#         labels={'pnl_pct': 'P&L (%)', 'profitable': 'Profitable'}
#     )
#     fig.update_layout(height=400)
#     st.plotly_chart(fig, use_container_width=True)

# # ============================================================================
# # VALIDATION PAGE
# # ============================================================================

# def render_validation_page(config):
#     """Render validation analysis page."""
#     st.markdown('<div class="main-header">✅ Strategy Validation</div>', unsafe_allow_html=True)
    
#     st.markdown("""
#     Comprehensive validation ensures your strategy is robust and not overfit to historical data.
#     This page runs industry-standard tests used by professional quant funds.
#     """)
    
#     if 'results' not in st.session_state:
#         st.warning("⚠️ Please run a backtest first from the 📊 Backtest tab")
#         return
    
#     st.markdown(f"""
#     **Validation Configuration:**
#     - Walk-Forward Folds: {config['n_folds']}
#     - Monte Carlo Simulations: {config['n_simulations']}
#     """)
    
#     # Run validation button
#     if st.button("🔍 Run Complete Validation", type="primary", use_container_width=True):
#         with st.spinner("Running comprehensive validation tests... This may take several minutes."):
#             try:
#                 validator = StrategyValidator(
#                     st.session_state['system'],
#                     st.session_state['results'],
#                     config['ticker_y'],
#                     config['ticker_x']
#                 )
                
#                 # Run all tests
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 # 1. Walk-Forward Analysis
#                 status_text.text("1/6: Running walk-forward analysis...")
#                 progress_bar.progress(0.17)
#                 wf_results = validator.walk_forward_analysis(
#                     st.session_state['stock_y'],
#                     st.session_state['stock_x'],
#                     st.session_state['market'],
#                     train_period=config['train_period'],
#                     test_period=126,
#                     n_folds=config['n_folds'],
#                     entry_z=config['entry_z'],
#                     exit_z=config['exit_z']
#                 )
                
#                 # 2. Monte Carlo Simulation
#                 status_text.text("2/6: Running Monte Carlo simulation...")
#                 progress_bar.progress(0.33)
#                 mc_results = validator.monte_carlo_simulation(n_simulations=config['n_simulations'])
                
#                 # 3. Stress Testing
#                 status_text.text("3/6: Running stress tests...")
#                 progress_bar.progress(0.50)
#                 stress_results = validator.stress_test_crisis_periods()
                
#                 # 4. Sensitivity Analysis
#                 status_text.text("4/6: Running sensitivity analysis...")
#                 progress_bar.progress(0.67)
#                 sensitivity_results = validator.sensitivity_analysis(
#                     st.session_state['stock_y'],
#                     st.session_state['stock_x'],
#                     st.session_state['market']
#                 )
                
#                 # 5. Transaction Cost Analysis
#                 status_text.text("5/6: Analyzing transaction costs...")
#                 progress_bar.progress(0.83)
#                 cost_results = validator.transaction_cost_analysis()
                
#                 # 6. Statistical Tests
#                 status_text.text("6/6: Running statistical tests...")
#                 progress_bar.progress(0.95)
#                 validator.statistical_tests()
                
#                 # Generate report
#                 status_text.text("Generating final report...")
#                 overall_score = validator.generate_report()
#                 progress_bar.progress(1.0)
                
#                 # Store in session state
#                 st.session_state['validator'] = validator
#                 st.session_state['validation_score'] = overall_score
                
#                 status_text.empty()
#                 progress_bar.empty()
#                 st.success("✅ Validation complete!")
                
#             except Exception as e:
#                 st.error(f"❌ Error during validation: {e}")
#                 return
    
#     # Display validation results if available
#     if 'validator' in st.session_state:
#         display_validation_results(st.session_state['validator'], st.session_state['validation_score'])

# def display_validation_results(validator, overall_score):
#     """Display comprehensive validation results."""
    
#     # Overall score
#     st.markdown('<div class="sub-header">🎯 Overall Validation Score</div>', unsafe_allow_html=True)
    
#     score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
    
#     with score_col2:
#         # Score gauge
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number+delta",
#             value=overall_score,
#             domain={'x': [0, 1], 'y': [0, 1]},
#             title={'text': "Validation Score", 'font': {'size': 24}},
#             delta={'reference': 80, 'increasing': {'color': "green"}},
#             gauge={
#                 'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
#                 'bar': {'color': "darkblue"},
#                 'bgcolor': "white",
#                 'borderwidth': 2,
#                 'bordercolor': "gray",
#                 'steps': [
#                     {'range': [0, 60], 'color': '#f8d7da'},
#                     {'range': [60, 80], 'color': '#fff3cd'},
#                     {'range': [80, 100], 'color': '#d4edda'}
#                 ],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 80
#                 }
#             }
#         ))
        
#         fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Recommendation
#         if overall_score >= 80:
#             st.markdown('<div class="success-box"><b>✅ STRATEGY VALIDATED</b><br>Strategy shows robust performance. Proceed with paper trading.</div>', unsafe_allow_html=True)
#         elif overall_score >= 60:
#             st.markdown('<div class="warning-box"><b>⚠️ MARGINAL PERFORMANCE</b><br>Strategy shows promise but has limitations. Use with caution.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="danger-box"><b>❌ STRATEGY NOT VALIDATED</b><br>Strategy fails critical tests. DO NOT TRADE.</div>', unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # Create tabs for each validation test
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "📊 Walk-Forward",
#         "🎲 Monte Carlo",
#         "⚠️ Stress Test",
#         "🔧 Sensitivity",
#         "💰 Transaction Costs"
#     ])
    
#     with tab1:
#         display_walk_forward_results(validator)
    
#     with tab2:
#         display_monte_carlo_results(validator)
    
#     with tab3:
#         display_stress_test_results(validator)
    
#     with tab4:
#         display_sensitivity_results(validator)
    
#     with tab5:
#         display_transaction_cost_results(validator)

# def display_walk_forward_results(validator):
#     """Display walk-forward analysis results."""
#     if 'walk_forward' not in validator.validation_results:
#         st.info("Walk-forward analysis not available")
#         return
    
#     st.markdown("### Walk-Forward Analysis (Out-of-Sample Testing)")
#     st.markdown("""
#     This test ensures the strategy wasn't overfit to historical data by training on one period 
#     and testing on the next period multiple times.
#     """)
    
#     wf_results = validator.validation_results['walk_forward']
    
#     # Summary metrics
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Avg OOS Return", f"{wf_results['Return'].mean():.2f}%")
#     col2.metric("Avg Sharpe", f"{wf_results['Sharpe'].mean():.2f}")
#     col3.metric("Consistency", f"{(wf_results['Return'] > 0).mean()*100:.0f}%")
#     col4.metric("Std Dev", f"{wf_results['Return'].std():.2f}%")
    
#     # Results table
#     st.markdown("#### Fold Results")
#     st.dataframe(wf_results, use_container_width=True, hide_index=True)
    
#     # Visualization
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=("Returns by Fold", "Sharpe Ratio by Fold"),
#         specs=[[{"type": "bar"}, {"type": "bar"}]]
#     )
    
#     fig.add_trace(go.Bar(
#         x=wf_results['Fold'],
#         y=wf_results['Return'],
#         marker_color=['#2A9D8F' if x > 0 else '#E76F51' for x in wf_results['Return']],
#         showlegend=False
#     ), row=1, col=1)
    
#     fig.add_trace(go.Bar(
#         x=wf_results['Fold'],
#         y=wf_results['Sharpe'],
#         marker_color=['#2A9D8F' if x > 0 else '#E76F51' for x in wf_results['Sharpe']],
#         showlegend=False
#     ), row=1, col=2)
    
#     fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
#     fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    
#     fig.update_xaxes(title_text="Fold", row=1, col=1)
#     fig.update_xaxes(title_text="Fold", row=1, col=2)
#     fig.update_yaxes(title_text="Return (%)", row=1, col=1)
#     fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    
#     fig.update_layout(height=400, showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)

# def display_monte_carlo_results(validator):
#     """Display Monte Carlo simulation results."""
#     if 'monte_carlo' not in validator.validation_results:
#         st.info("Monte Carlo simulation not available")
#         return
    
#     st.markdown("### Monte Carlo Simulation")
#     st.markdown("""
#     This test randomly shuffles trade returns to determine if the strategy's performance 
#     is statistically significant or due to luck.
#     """)
    
#     mc = validator.validation_results['monte_carlo']
    
#     # Summary metrics
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Actual Return", f"{mc['actual_return']*100:.2f}%")
#     col2.metric("P-Value", f"{mc['p_value']:.4f}")
#     col3.metric("95% CI Lower", f"{mc['ci_lower']*100:.2f}%")
#     col4.metric("95% CI Upper", f"{mc['ci_upper']*100:.2f}%")
    
#     # Distribution plot
#     fig = go.Figure()
    
#     fig.add_trace(go.Histogram(
#         x=mc['simulated_returns'] * 100,
#         nbinsx=50,
#         name='Simulated Returns',
#         marker_color='#264653',
#         opacity=0.7
#     ))
    
#     fig.add_vline(
#         x=mc['actual_return'] * 100,
#         line_dash="dash",
#         line_color="red",
#         line_width=3,
#         annotation_text="Actual Return",
#         annotation_position="top"
#     )
    
#     fig.add_vline(
#         x=mc['ci_lower'] * 100,
#         line_dash="dot",
#         line_color="green",
#         annotation_text="95% CI"
#     )
    
#     fig.add_vline(
#         x=mc['ci_upper'] * 100,
#         line_dash="dot",
#         line_color="green"
#     )
    
#     fig.update_layout(
#         title="Monte Carlo Simulation Results",
#         xaxis_title="Return (%)",
#         yaxis_title="Frequency",
#         height=500,
#         showlegend=False
#     )
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Interpretation
#     if mc['p_value'] < 0.05:
#         st.success("✅ Returns are statistically significant (p < 0.05)")
#     elif mc['p_value'] < 0.10:
#         st.warning("⚠️ Returns show marginal significance (p < 0.10)")
#     else:
#         st.error("❌ Returns not statistically significant - may be due to luck")

# def display_stress_test_results(validator):
#     """Display stress test results."""
#     if 'stress_test' not in validator.validation_results:
#         st.info("Stress test results not available")
#         return
    
#     st.markdown("### Stress Testing (Crisis Periods)")
#     st.markdown("""
#     This test evaluates how the strategy performs during known market crises and high volatility periods.
#     """)
    
#     stress = validator.validation_results['stress_test']
    
#     if stress is not None and len(stress) > 0:
#         # Results table
#         st.dataframe(stress, use_container_width=True, hide_index=True)
        
#         # Visualization
#         fig = make_subplots(
#             rows=1, cols=2,
#             subplot_titles=("Strategy vs Market Returns", "Trading Activity During Crises")
#         )
        
#         fig.add_trace(go.Bar(
#             x=stress['Crisis'],
#             y=stress['Strategy_Return'],
#             name='Strategy',
#             marker_color='#2A9D8F'
#         ), row=1, col=1)
        
#         fig.add_trace(go.Bar(
#             x=stress['Crisis'],
#             y=stress['Market_Return'],
#             name='Market',
#             marker_color='#E76F51'
#         ), row=1, col=1)
        
#         fig.add_trace(go.Bar(
#             x=stress['Crisis'],
#             y=stress['Trade_Rate'],
#             marker_color='#264653',
#             showlegend=False
#         ), row=1, col=2)
        
#         fig.update_xaxes(title_text="Crisis Period", row=1, col=1)
#         fig.update_xaxes(title_text="Crisis Period", row=1, col=2)
#         fig.update_yaxes(title_text="Return (%)", row=1, col=1)
#         fig.update_yaxes(title_text="Trading Rate (%)", row=1, col=2)
        
#         fig.update_layout(height=500)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("No crisis periods found in the backtest timeframe")

# def display_sensitivity_results(validator):
#     """Display sensitivity analysis results."""
#     if 'sensitivity' not in validator.validation_results:
#         st.info("Sensitivity analysis not available")
#         return
    
#     st.markdown("### Sensitivity Analysis (Parameter Robustness)")
#     st.markdown("""
#     This test shows how the strategy performs across different parameter settings to ensure 
#     it's not overfit to specific values.
#     """)
    
#     sens = validator.validation_results['sensitivity']
    
#     # Results table
#     st.dataframe(sens, use_container_width=True, hide_index=True)
    
#     # Heatmap
#     pivot_return = sens.pivot(index='entry_z', columns='exit_z', values='Return_%')
#     pivot_sharpe = sens.pivot(index='entry_z', columns='exit_z', values='Sharpe')
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         fig = go.Figure(data=go.Heatmap(
#             z=pivot_return.values,
#             x=pivot_return.columns,
#             y=pivot_return.index,
#             colorscale='RdYlGn',
#             text=pivot_return.values,
#             texttemplate='%{text:.1f}%',
#             textfont={"size": 10},
#             colorbar=dict(title="Return (%)")
#         ))
#         fig.update_layout(
#             title="Returns Heatmap",
#             xaxis_title="Exit Z-Score",
#             yaxis_title="Entry Z-Score",
#             height=400
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         fig = go.Figure(data=go.Heatmap(
#             z=pivot_sharpe.values,
#             x=pivot_sharpe.columns,
#             y=pivot_sharpe.index,
#             colorscale='RdYlGn',
#             text=pivot_sharpe.values,
#             texttemplate='%{text:.2f}',
#             textfont={"size": 10},
#             colorbar=dict(title="Sharpe")
#         ))
#         fig.update_layout(
#             title="Sharpe Ratio Heatmap",
#             xaxis_title="Exit Z-Score",
#             yaxis_title="Entry Z-Score",
#             height=400
#         )
#         st.plotly_chart(fig, use_container_width=True)

# def display_transaction_cost_results(validator):
#     """Display transaction cost analysis results."""
#     if 'transaction_costs' not in validator.validation_results:
#         st.info("Transaction cost analysis not available")
#         return
    
#     st.markdown("### Transaction Cost Analysis")
#     st.markdown("""
#     This test applies realistic trading costs (slippage and commissions) to see if the 
#     strategy remains profitable after accounting for implementation costs.
#     """)
    
#     tc = validator.validation_results['transaction_costs']
    
#     # Metrics
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Original Return", f"{tc['original_return']:.2f}%")
#     col2.metric("After Costs", f"{tc['return_after_costs']:.2f}%")
#     col3.metric("Cost Impact", f"{tc['cost_impact']:.2f}%")
    
#     # Visualization
#     fig = go.Figure()
    
#     categories = ['Original Return', 'Transaction Costs', 'Net Return']
#     values = [
#         tc['original_return'],
#         -tc['cost_impact'],
#         tc['return_after_costs']
#     ]
#     colors = ['#2A9D8F', '#E76F51', '#264653']
    
#     fig.add_trace(go.Bar(
#         x=categories,
#         y=values,
#         marker_color=colors,
#         text=[f"{v:.2f}%" for v in values],
#         textposition='auto'
#     ))
    
#     fig.add_hline(y=0, line_dash="dash", line_color="black")
    
#     fig.update_layout(
#         title="Impact of Transaction Costs",
#         yaxis_title="Return (%)",
#         height=400,
#         showlegend=False
#     )
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Retention rate
#     retention = tc['return_after_costs'] / tc['original_return'] if tc['original_return'] != 0 else 0
#     st.metric("Return Retention", f"{retention*100:.1f}%")
    
#     if retention > 0.7:
#         st.success("✅ Strategy retains >70% of returns after costs")
#     elif retention > 0.5:
#         st.warning("⚠️ Strategy retains 50-70% of returns after costs")
#     else:
#         st.error("❌ Strategy loses >50% of returns to transaction costs")

# # ============================================================================
# # LIVE ANALYSIS PAGE
# # ============================================================================

# def render_live_analysis_page(config):
#     """Render live analysis page."""
#     st.markdown('<div class="main-header">📈 Live Analysis</div>', unsafe_allow_html=True)
    
#     st.info("🚧 Live trading features coming soon! This will include real-time signals and position monitoring.")
    
#     if 'results' not in st.session_state:
#         st.warning("⚠️ Please run a backtest first to see current signals")
#         return
    
#     results = st.session_state['results']
#     system = st.session_state['system']
    
#     # Current status
#     st.markdown('<div class="sub-header">📊 Current Status</div>', unsafe_allow_html=True)
    
#     latest = results.iloc[-1]
    
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Current Z-Score", f"{latest['z_score']:.2f}")
#     col2.metric("Current Signal", ["Flat", "Long", "Short"][int(latest['final_signal'])+1])
#     col3.metric("Regime", "SAFE ✅" if latest['regime'] == 1 else "UNSAFE ⚠️")
#     col4.metric("Hedge Ratio", f"{latest['hedge_ratio']:.4f}")
    
#     # Recent signals
#     st.markdown("#### Recent Signals (Last 30 Days)")
#     recent = results.tail(30)[['z_score', 'math_signal', 'regime', 'final_signal']]
#     recent.columns = ['Z-Score', 'Math Signal', 'Safe Regime', 'Final Signal']
#     st.dataframe(recent, use_container_width=True)

# # ============================================================================
# # DOCUMENTATION PAGE
# # ============================================================================

# def render_documentation_page():
#     """Render documentation page."""
#     st.markdown('<div class="main-header">📚 Documentation</div>', unsafe_allow_html=True)
    
#     st.markdown("""
#     ## System Overview
    
#     The Hybrid Pairs Trading System combines mathematical precision with machine learning intelligence 
#     to trade cointegrated stock pairs while avoiding dangerous market conditions.
    
#     ### 🧮 Component 1: Kalman Filter (Mathematical Core)
    
#     The Kalman Filter is a recursive algorithm that dynamically estimates the hedge ratio between two stocks:
    
#     - **Purpose**: Track the relationship between Stock Y and Stock X
#     - **Output**: Spread (Y - hedge_ratio × X) and its standard deviation
#     - **Advantage**: Adapts to changing market conditions in real-time
    
#     #### Z-Score Calculation
    
#     ```
#     z_score = (spread - rolling_mean) / rolling_std
#     ```
    
#     - Entry: |z_score| > 2.0 (spread is extreme)
#     - Exit: |z_score| < 0.5 (spread has reverted)
    
#     ### 🤖 Component 2: Random Forest Classifier (ML Regime Filter)
    
#     The ML model classifies market conditions as SAFE or UNSAFE:
    
#     **Features used:**
#     - Market volatility (20-day and 5-day)
#     - Maximum drawdown (60-day)
#     - Pair correlation changes
#     - Volume ratios (if available)
    
#     **Labels:**
#     - SAFE (1): Normal market conditions, trade as usual
#     - UNSAFE (0): High volatility or drawdown, avoid trading
    
#     ### ⚡ Hybrid Logic
    
#     The final trading decision combines both components:
    
#     ```
#     Final Signal = Math Signal × Regime Filter
#     ```
    
#     This means:
#     - **Trade**: Only when math gives a signal AND market is safe
#     - **Stay Flat**: If either condition is not met
    
#     ### 📊 Performance Metrics
    
#     #### Sharpe Ratio
#     ```
#     Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility
#     ```
#     Higher is better. Above 1.0 is good, above 2.0 is excellent.
    
#     #### Maximum Drawdown
#     Maximum peak-to-trough decline during the period. Lower (less negative) is better.
    
#     #### Win Rate
#     Percentage of profitable trades. 55-60% is typical for pairs trading.
    
#     ### ✅ Validation Tests
    
#     #### 1. Walk-Forward Analysis
#     - Train on period N, test on period N+1
#     - Repeat multiple times (folds)
#     - Ensures strategy works out-of-sample
    
#     #### 2. Monte Carlo Simulation
#     - Randomly shuffle trade returns 1000+ times
#     - Calculate p-value: probability results are due to chance
#     - p < 0.05 means statistically significant
    
#     #### 3. Stress Testing
#     - Test during known crisis periods (COVID crash, bear markets)
#     - Strategy should either profit or avoid trading
    
#     #### 4. Sensitivity Analysis
#     - Test multiple parameter combinations
#     - Strategy should work across a range of values
#     - Avoid strategies that only work with one specific parameter set
    
#     #### 5. Transaction Cost Analysis
#     - Apply realistic slippage (5 bps typical)
#     - Ensure strategy remains profitable after costs
    
#     ### 🎯 Interpretation Guide
    
#     **Overall Validation Score:**
#     - 80-100: ✅ VALIDATED - Ready for paper trading
#     - 60-80: ⚠️ MARGINAL - Use with caution
#     - 0-60: ❌ FAILED - Do not trade
    
#     **Walk-Forward:**
#     - Pass: >60% of folds profitable, avg Sharpe > 0.5
#     - Fail: Inconsistent or negative returns out-of-sample
    
#     **Monte Carlo:**
#     - Pass: p-value < 0.05
#     - Fail: p-value > 0.10 (results likely due to luck)
    
#     ### ⚠️ Important Warnings
    
#     1. **Past performance ≠ future results**
#     2. **Always start with paper trading**
#     3. **Use proper position sizing (1-5% of capital per trade)**
#     4. **Monitor correlation breakdown** - if pairs diverge permanently, exit
#     5. **This is for educational purposes only**
    
#     ### 📖 Further Reading
    
#     - **Pairs Trading**: Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006)
#     - **Kalman Filter**: Welch, G., & Bishop, G. (2006)
#     - **Statistical Arbitrage**: Avellaneda, M., & Lee, J. H. (2010)
#     - **Regime Switching**: Hamilton, J. D. (1989)
#     """)
    
#     # Code example
#     st.markdown("### 💻 Code Example")
    
#     code = """
# # Basic usage example
# from enhanced_pairs_trading import HybridPairsTradingSystem, download_data

# # 1. Download data
# stock_y, stock_x, market = download_data('PEP', 'KO', 'SPY', '2018-01-01', '2024-01-01')

# # 2. Initialize system
# system = HybridPairsTradingSystem()

# # 3. Run backtest
# results = system.run_backtest(stock_y, stock_x, market, entry_z=2.0, exit_z=0.5)

# # 4. Get performance metrics
# metrics = system.get_performance_metrics()
# print(metrics)

# # 5. Analyze trades
# trades = system.get_trade_analysis()
# print(f"Total trades: {len(trades)}")
# print(f"Win rate: {trades['profitable'].mean()*100:.1f}%")
# """
    
#     st.code(code, language='python')

# # ============================================================================
# # MAIN APPLICATION
# # ============================================================================

# def main():
#     """Main application entry point."""
    
#     # Check imports
#     if not IMPORTS_SUCCESS:
#         st.error(f"❌ Failed to import required modules: {IMPORT_ERROR}")
#         st.info("""
#         **Make sure you have these files in the same directory:**
#         - `enhanced_pairs_trading.py` - Main trading system
#         - `strategy_validator.py` - Validation framework
#         - `streamlit_app.py` - This file
        
#         **Install required packages:**
#         ```bash
#         pip install streamlit pandas numpy scipy scikit-learn yfinance matplotlib plotly seaborn
#         ```
#         """)
#         return
    
#     # Render sidebar and get configuration
#     config = render_sidebar()
    
#     # Route to appropriate page
#     page = config['page']
    
#     if page == "🏠 Home":
#         render_home_page()
#     elif page == "📊 Backtest":
#         render_backtest_page(config)
#     elif page == "✅ Validation":
#         render_validation_page(config)
#     elif page == "📈 Live Analysis":
#         render_live_analysis_page(config)
#     elif page == "📚 Documentation":
#         render_documentation_page()
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center; color: #666; padding: 2rem;'>
#         <p>Hybrid Pairs Trading System v1.0</p>
#         <p>⚠️ For educational purposes only. Not financial advice.</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()


# Adaptive Kalman filter App.py

# """
# Enhanced Streamlit Dashboard for Hybrid Pairs Trading System
# Compatible with Adaptive Kalman Filter version
# Run with: streamlit run dashboard.py
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.express as px
# from datetime import datetime, timedelta

# # Import enhanced trading system
# from hybrid_pairs_trading import (
#     HybridPairsTradingSystem,
#     download_data,
#     AdaptiveKalmanPairs,
#     MarketRegimeClassifier
# )

# # Page configuration
# st.set_page_config(
#     page_title="Enhanced Hybrid Pairs Trading",
#     page_icon="",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #2E86AB;
#     }
#     .positive { color: #28a745; }
#     .negative { color: #dc3545; }
#     .info-box {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 1rem 0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Title
# st.markdown('<p class="main-header">Enhanced Hybrid Pairs Trading System</p>', unsafe_allow_html=True)
# st.markdown("---")

# # ============================================================================
# # SIDEBAR CONFIGURATION
# # ============================================================================

# st.sidebar.header("Configuration")

# # Pair Selection
# st.sidebar.subheader("Select Trading Pair")
# pair_presets = {
#     "PEP vs KO (Beverages)": ("PEP", "KO"),
#     "JPM vs BAC (Banks)": ("JPM", "BAC"),
#     "XOM vs CVX (Energy)": ("XOM", "CVX"),
#     "MSFT vs ORCL (Tech)": ("MSFT", "ORCL"),
#     "GLD vs GDX (Gold)": ("GLD", "GDX"),
#     "Custom": ("CUSTOM", "CUSTOM")
# }

# selected_pair = st.sidebar.selectbox(
#     "Choose pair:",
#     options=list(pair_presets.keys())
# )

# if selected_pair == "Custom":
#     ticker_y = st.sidebar.text_input("Stock Y (Dependent):", "PEP")
#     ticker_x = st.sidebar.text_input("Stock X (Independent):", "KO")
# else:
#     ticker_y, ticker_x = pair_presets[selected_pair]

# # Date Range
# st.sidebar.subheader("Date Range")
# col1, col2 = st.sidebar.columns(2)
# start_date = col1.date_input("Start Date", datetime(2018, 1, 1))
# end_date = col2.date_input("End Date", datetime(2024, 1, 1))

# # Strategy Parameters
# st.sidebar.subheader("Trading Parameters")

# entry_z = st.sidebar.slider(
#     "Entry Z-Score Threshold",
#     min_value=1.0, max_value=3.0, value=2.0, step=0.1,
#     help="Z-score threshold to enter positions (recommended: 1.8-2.5)"
# )

# exit_z = st.sidebar.slider(
#     "Exit Z-Score Threshold",
#     min_value=0.1, max_value=1.0, value=0.5, step=0.1,
#     help="Z-score threshold to exit positions (recommended: 0.3-0.7)"
# )

# train_period = st.sidebar.number_input(
#     "Training Period (days)",
#     min_value=100, max_value=500, value=252, step=50,
#     help="Number of days to train ML regime classifier"
# )

# # Advanced Parameters (Collapsible)
# with st.sidebar.expander(" Advanced Kalman Filters"):
#     st.markdown("**Adaptive Kalman Filter Settings**")
    
#     Q_init = st.number_input(
#         "Q_init (Process Noise)",
#         min_value=1e-6, max_value=1e-3, value=1e-5, format="%.6f",
#         help="Initial process noise (lower = more stable)"
#     )
    
#     R = st.number_input(
#         "R (Observation Noise)",
#         min_value=0.1, max_value=10.0, value=1.0, step=0.1,
#         help="Observation noise (adjust based on price scale)"
#     )
    
#     lambda_Q = st.slider(
#         "λ_Q (Q Decay Factor)",
#         min_value=0.950, max_value=0.999, value=0.995, step=0.001,
#         help="Higher = smoother Q adaptation (0.98-0.999)"
#     )
    
#     adapt_scale = st.slider(
#         "Adaptation Scale",
#         min_value=0.1, max_value=2.0, value=0.5, step=0.1,
#         help="Scaling factor for Q adaptation speed"
#     )
    
#     ewma_alpha = st.slider(
#         "EWMA Alpha",
#         min_value=0.01, max_value=0.15, value=0.03, step=0.01,
#         help="EWMA smoothing (lower = slower, 0.01-0.1)"
#     )
    
#     ema_short_span = st.number_input(
#         "EMA Short Span",
#         min_value=3, max_value=10, value=5, step=1,
#         help="Short EMA period (3-7 typical)"
#     )
    
#     ema_long_span = st.number_input(
#         "EMA Long Span",
#         min_value=10, max_value=30, value=20, step=5,
#         help="Long EMA period (15-25 typical)"
#     )
    
#     use_ema_confirmation = st.checkbox(
#         "Use EMA Crossover Confirmation",
#         value=True,
#         help="Require EMA crossover to confirm entry signals"
#     )

# # Run Backtest Button
# st.sidebar.markdown("---")
# run_button = st.sidebar.button(" Backtest", type="primary", use_container_width=True)


# # Initialize session state
# if 'results' not in st.session_state:
#     st.session_state.results = None
# if 'system' not in st.session_state:
#     st.session_state.system = None
# if 'metrics' not in st.session_state:
#     st.session_state.metrics = None

# # ============================================================================
# # RUN BACKTEST
# # ============================================================================

# if run_button:
#     with st.spinner(f"Running enhanced backtest for {ticker_y} vs {ticker_x}..."):
#         try:
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Download data
#             status_text.info(f"Downloading data for {ticker_y}, {ticker_x}, and SPY...")
#             progress_bar.progress(20)
            
#             stock_y, stock_x, spy = download_data(
#                 ticker_y, ticker_x, 'SPY',
#                 start_date.strftime('%Y-%m-%d'),
#                 end_date.strftime('%Y-%m-%d')
#             )
            
#             progress_bar.progress(40)
            
#             # Initialize enhanced system
#             status_text.info(" Initializing ")
#             system = HybridPairsTradingSystem()
#             progress_bar.progress(50)
            
#             # Run backtest with enhanced parameters
#             status_text.info(" Running backtest ")
#             results = system.run_backtest(
#                 stock_y=stock_y,
#                 stock_x=stock_x,
#                 market_index=spy,
#                 train_period=train_period,
#                 entry_z=entry_z,
#                 exit_z=exit_z,
#                 Q_init=Q_init,
#                 R=R,
#                 lambda_Q=lambda_Q,
#                 adapt_scale=adapt_scale,
#                 ewma_alpha=ewma_alpha,
#                 ema_short_span=ema_short_span,
#                 ema_long_span=ema_long_span,
#                 use_ema_confirmation=use_ema_confirmation
#             )
#             progress_bar.progress(80)
            
#             # Get metrics
#             status_text.info("Calculating performance ")
#             metrics = system.get_performance_metrics()
#             progress_bar.progress(100)
            
#             # Store in session state
#             st.session_state.results = results
#             st.session_state.system = system
#             st.session_state.metrics = metrics
#             st.session_state.ticker_y = ticker_y
#             st.session_state.ticker_x = ticker_x
#             st.session_state.params = {
#                 'entry_z': entry_z,
#                 'exit_z': exit_z,
#                 'Q_init': Q_init,
#                 'R': R,
#                 'lambda_Q': lambda_Q,
#                 'ewma_alpha': ewma_alpha,
#                 'ema_short_span': ema_short_span,
#                 'ema_long_span': ema_long_span
#             }
            
#             progress_bar.empty()
#             status_text.empty()
            
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")
#             st.exception(e)
#             st.stop()

# # ============================================================================
# # DISPLAY RESULTS
# # ============================================================================

# if st.session_state.results is not None:
#     results = st.session_state.results
#     system = st.session_state.system
#     metrics = st.session_state.metrics
#     ticker_y = st.session_state.ticker_y
#     ticker_x = st.session_state.ticker_x
#     params = st.session_state.params

    
#     # Tabs
#     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#         " Overview", 
#         " Performance", 
#         " Kalman Analysis",
#         " Signals", 
#         " Risk Analysis",
#         " Trade Log"
#     ])
    
#     # TAB 1: OVERVIEW
#     with tab1:
#         st.header("Performance Overview")
        
#         hybrid_metrics = metrics['Hybrid Strategy']
#         math_metrics = metrics['Math Only']
        
#         # Key Metrics Row
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             hybrid_ret = float(hybrid_metrics['Total Return'].rstrip('%'))
#             math_ret = float(math_metrics['Total Return'].rstrip('%'))
#             st.metric(
#                 label=" Total Return (Hybrid)",
#                 value=hybrid_metrics['Total Return'],
#                 delta=f"{hybrid_ret - math_ret:+.2f}% vs Math Only"
#             )
        
#         with col2:
#             hybrid_sharpe = float(hybrid_metrics['Sharpe Ratio'])
#             math_sharpe = float(math_metrics['Sharpe Ratio'])
#             st.metric(
#                 label=" Sharpe Ratio ",
#                 value=hybrid_metrics['Sharpe Ratio'],
#                 delta=f"{hybrid_sharpe - math_sharpe:+.2f} vs Math Only"
#             )
        
#         with col3:
#             st.metric(
#                 label=" Max Drawdown ",
#                 value=hybrid_metrics['Max Drawdown'],
#                 delta=f"vs Math: {math_metrics['Max Drawdown']}",
#                 delta_color="inverse"
#             )
        
#         with col4:
#             st.metric(
#                 label=" Win Rate ",
#                 value=hybrid_metrics['Win Rate'],
#                 delta=f"{hybrid_metrics['Total Trades']} trades"
#             )
        
#         st.markdown("---")
        
#         # Cumulative Returns Chart
#         st.subheader(" Cumulative Returns Comparison")
        
#         fig_returns = go.Figure()
        
#         fig_returns.add_trace(go.Scatter(
#             x=results.index,
#             y=(results['strategy_cumulative'] - 1) * 100,
#             name='Hybrid (Adaptive KF + ML)',
#             line=dict(color='#2A9D8F', width=3),
#             hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
#         ))
        
#         fig_returns.add_trace(go.Scatter(
#             x=results.index,
#             y=(results['math_only_cumulative'] - 1) * 100,
#             name='Math Only (No Filter)',
#             line=dict(color='#E76F51', width=2, dash='dash'),
#             hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
#         ))
        
#         fig_returns.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
#         fig_returns.update_layout(
#             height=450,
#             hovermode='x unified',
#             xaxis_title="Date",
#             yaxis_title="Cumulative Return (%)",
#             legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
#             template="plotly_white"
#         )
        
#         st.plotly_chart(fig_returns, use_container_width=True)
        
#         # Side-by-side comparison
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Hybrid Strategy Metrics")
           
#             for key, value in hybrid_metrics.items():
#                 st.metric(label=key, value=value)
        
#         with col2:
#             st.subheader(" Math Only Metrics")
            
#             for key, value in math_metrics.items():
#                 st.metric(label=key, value=value)
        
#         # Additional Info
#         st.markdown("---")
#         st.subheader("Session Information")
#         info_metrics = metrics['Additional Info']
        
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("Total Days", info_metrics['Total Trading Days'])
#         col2.metric("Safe Regime Days", info_metrics['Safe Regime Days'])
#         col3.metric("Unsafe Days", info_metrics['Unsafe Regime Days'])
#         col4.metric("Filter Rate", info_metrics['Regime Filter Rate'])
    
#     # TAB 2: PERFORMANCE
#     with tab2:
#         st.header("Detailed Performance Analysis")
        
#         # Normalized Prices
#         st.subheader(f" Stock Prices: {ticker_y} vs {ticker_x}")
        
#         fig_prices = go.Figure()
        
#         y_norm = (results['Y'] / results['Y'].iloc[0]) * 100
#         x_norm = (results['X'] / results['X'].iloc[0]) * 100
        
#         fig_prices.add_trace(go.Scatter(
#             x=results.index, y=y_norm, name=ticker_y,
#             line=dict(color='#2E86AB', width=2)
#         ))
        
#         fig_prices.add_trace(go.Scatter(
#             x=results.index, y=x_norm, name=ticker_x,
#             line=dict(color='#A23B72', width=2)
#         ))
        
#         # Shade unsafe regimes
#         unsafe = results[results['regime'] == 0]
#         if len(unsafe) > 0:
#             unsafe_groups = (unsafe.index.to_series().diff() > pd.Timedelta(days=2)).cumsum()
#             for group_id in unsafe_groups.unique():
#                 group = unsafe[unsafe_groups == group_id]
#                 fig_prices.add_vrect(
#                     x0=group.index[0], x1=group.index[-1],
#                     fillcolor="red", opacity=0.1,
#                     layer="below", line_width=0
#                 )
        
#         fig_prices.update_layout(
#             height=400, hovermode='x unified',
#             xaxis_title="Date", yaxis_title="Normalized Price (Base=100)",
#             template="plotly_white"
#         )
        
#         st.plotly_chart(fig_prices, use_container_width=True)
        
#         # Drawdown Analysis
#         st.subheader(" Drawdown Analysis")
        
#         hybrid_cum = results['strategy_cumulative']
#         math_cum = results['math_only_cumulative']
        
#         hybrid_dd = ((hybrid_cum - hybrid_cum.expanding().max()) / hybrid_cum.expanding().max()) * 100
#         math_dd = ((math_cum - math_cum.expanding().max()) / math_cum.expanding().max()) * 100
        
#         fig_dd = go.Figure()
        
#         fig_dd.add_trace(go.Scatter(
#             x=results.index, y=hybrid_dd, name='Hybrid',
#             fill='tozeroy', line=dict(color='#2A9D8F', width=0),
#             fillcolor='rgba(42, 157, 143, 0.5)'
#         ))
        
#         fig_dd.add_trace(go.Scatter(
#             x=results.index, y=math_dd, name='Math Only',
#             fill='tozeroy', line=dict(color='#E76F51', width=0),
#             fillcolor='rgba(231, 111, 81, 0.3)'
#         ))
        
#         fig_dd.update_layout(
#             height=350, hovermode='x unified',
#             xaxis_title="Date", yaxis_title="Drawdown (%)",
#             template="plotly_white"
#         )
        
#         st.plotly_chart(fig_dd, use_container_width=True)
        
#         # Rolling Metrics
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader(" Rolling Sharpe (30-day)")
#             rolling_sharpe = (
#                 results['strategy_return'].rolling(30).mean() / 
#                 results['strategy_return'].rolling(30).std() * np.sqrt(252)
#             )
            
#             fig_sharpe = go.Figure()
#             fig_sharpe.add_trace(go.Scatter(
#                 x=results.index, y=rolling_sharpe,
#                 line=dict(color='#2E86AB', width=2)
#             ))
#             fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="green")
#             fig_sharpe.add_hline(y=0.0, line_dash="dash", line_color="red")
#             fig_sharpe.update_layout(height=300, template="plotly_white", showlegend=False)
#             st.plotly_chart(fig_sharpe, use_container_width=True)
        
#         with col2:
#             st.subheader(" Rolling Returns (30-day)")
#             rolling_returns = results['strategy_return'].rolling(30).sum() * 100
            
#             fig_roll_ret = go.Figure()
#             fig_roll_ret.add_trace(go.Scatter(
#                 x=results.index, y=rolling_returns,
#                 line=dict(color='#A23B72', width=2), fill='tozeroy'
#             ))
#             fig_roll_ret.add_hline(y=0, line_dash="dot", line_color="gray")
#             fig_roll_ret.update_layout(height=300, template="plotly_white", showlegend=False)
#             st.plotly_chart(fig_roll_ret, use_container_width=True)
    
#     # TAB 3: KALMAN ANALYSIS
#     with tab3:
#         st.header(" Adaptive Kalman Filter Analysis")
        
        
#         # Adaptive Q Evolution
#         st.subheader(" Noise Processing(Q) Evolution")
#         st.markdown("""
#         **Process Noise (Q)** controls how quickly the filter adapts:
#         - **Higher Q** = Filter responds faster (during regime changes)
#         - **Lower Q** = Filter is more stable (during calm periods)
#         - The adaptive mechanism automatically adjusts Q based on market conditions
#         """)
        
#         fig_Q = go.Figure()
#         fig_Q.add_trace(go.Scatter(
#             x=results.index,
#             y=results['Q_adaptive'],
#             line=dict(color='#F4A261', width=2),
#             fill='tozeroy',
#             fillcolor='rgba(244, 162, 97, 0.3)',
#             name='Adaptive Q'
#         ))
        
#         fig_Q.update_layout(
#             height=300,
#             xaxis_title="Date",
#             yaxis_title="Q (log scale)",
#             yaxis_type="log",
#             template="plotly_white",
#             showlegend=False
#         )
        
#         st.plotly_chart(fig_Q, use_container_width=True)
        
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Average Q", f"{results['Q_adaptive'].mean():.2e}")
#         col2.metric("Max Q", f"{results['Q_adaptive'].max():.2e}")
#         col3.metric("Min Q", f"{results['Q_adaptive'].min():.2e}")
        
#         st.markdown("---")
        
#         # Hedge Ratio Evolution
#         st.subheader(" Dynamic Hedge Ratio Evolution (β)")
#         st.markdown(f"""
#         **Hedge Ratio (β)** represents the relationship: `{ticker_y} = β × {ticker_x} + spread`
#         - Current β = {results['beta_kf'].iloc[-1]:.4f}
#         - Interpretation: 1 share of {ticker_y} ≈ {results['beta_kf'].iloc[-1]:.4f} shares of {ticker_x}
#         """)
        
#         fig_beta = go.Figure()
#         fig_beta.add_trace(go.Scatter(
#             x=results.index,
#             y=results['beta_kf'],
#             line=dict(color='#2A9D8F', width=2),
#             fill='tozeroy',
#             fillcolor='rgba(42, 157, 143, 0.2)',
#             name='Hedge Ratio'
#         ))
        
#         fig_beta.update_layout(
#             height=300,
#             xaxis_title="Date",
#             yaxis_title="β (Hedge Ratio)",
#             template="plotly_white",
#             showlegend=False
#         )
        
#         st.plotly_chart(fig_beta, use_container_width=True)
        
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Current β", f"{results['beta_kf'].iloc[-1]:.4f}")
#         col2.metric("Average β", f"{results['beta_kf'].mean():.4f}")
#         col3.metric("β Std Dev", f"{results['beta_kf'].std():.4f}")
        
#         st.markdown("---")
        
#         # EMA Crossovers
#         st.subheader("EMA Short/Long Crossovers")
#         st.markdown("""
#         **EMA Confirmation** reduces false signals:
#         - **Green markers** = Bullish crossover (EMA short crosses above long)
#         - **Red markers** = Bearish crossover (EMA short crosses below long)
#         - Entries only occur when EMA confirms the z-score signal
#         """)
        
#         fig_ema = go.Figure()
        
#         fig_ema.add_trace(go.Scatter(
#             x=results.index,
#             y=results['ema_s'],
#             name=f'EMA Short ({params["ema_short_span"]})',
#             line=dict(color='#E9C46A', width=2)
#         ))
        
#         fig_ema.add_trace(go.Scatter(
#             x=results.index,
#             y=results['ema_l'],
#             name=f'EMA Long ({params["ema_long_span"]})',
#             line=dict(color='#E76F51', width=2)
#         ))
        
#         # Detect crossovers
#         cross_up = (results['ema_s'] > results['ema_l']) & (results['ema_s'].shift(1) <= results['ema_l'].shift(1))
#         cross_down = (results['ema_s'] < results['ema_l']) & (results['ema_s'].shift(1) >= results['ema_l'].shift(1))
        
#         if cross_up.sum() > 0:
#             fig_ema.add_trace(go.Scatter(
#                 x=results.index[cross_up],
#                 y=results['ema_s'][cross_up],
#                 mode='markers',
#                 name='Bullish Cross',
#                 marker=dict(color='green', size=10, symbol='triangle-up')
#             ))
        
#         if cross_down.sum() > 0:
#             fig_ema.add_trace(go.Scatter(
#                 x=results.index[cross_down],
#                 y=results['ema_s'][cross_down],
#                 mode='markers',
#                 name='Bearish Cross',
#                 marker=dict(color='red', size=10, symbol='triangle-down')
#             ))
        
#         fig_ema.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
#         fig_ema.update_layout(
#             height=350,
#             xaxis_title="Date",
#             yaxis_title="Spread",
#             template="plotly_white",
#             hovermode='x unified'
#         )
        
#         st.plotly_chart(fig_ema, use_container_width=True)
        
#         col1, col2 = st.columns(2)
#         col1.metric("Bullish Crossovers", cross_up.sum())
#         col2.metric("Bearish Crossovers", cross_down.sum())
    
#     # TAB 4: SIGNALS
#     with tab4:
#         st.header(" Trading Signals Analysis")
        
#         # Z-Score with EWMA
#         st.subheader(" Z-Score  & Trading Signals")
        
#         fig_zscore = make_subplots(
#             rows=2, cols=1,
#             row_heights=[0.6, 0.4],
#             subplot_titles=("Spread Z-Score", "Trading Signals"),
#             vertical_spacing=0.1
#         )
        
#         # Z-Score plot
#         fig_zscore.add_trace(
#             go.Scatter(
#                 x=results.index,
#                 y=results['z_ema'],
#                 name='Z-Score (EWMA)',
#                 line=dict(color='#264653', width=2)
#             ),
#             row=1, col=1
#         )
        
#         # Threshold lines
#         fig_zscore.add_hline(y=params['entry_z'], line_dash="dash", line_color="red", row=1, col=1)
#         fig_zscore.add_hline(y=-params['entry_z'], line_dash="dash", line_color="red", row=1, col=1)
#         fig_zscore.add_hline(y=params['exit_z'], line_dash="dot", line_color="green", row=1, col=1)
#         fig_zscore.add_hline(y=-params['exit_z'], line_dash="dot", line_color="green", row=1, col=1)
#         fig_zscore.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=1, col=1)
        
#         # Signals plot
#         fig_zscore.add_trace(
#             go.Scatter(
#                 x=results.index,
#                 y=results['math_signal'],
#                 name='Math Signal',
#                 fill='tozeroy',
#                 line=dict(width=0),
#                 fillcolor='rgba(128, 128, 128, 0.3)'
#             ),
#             row=2, col=1
#         )
        
#         fig_zscore.add_trace(
#             go.Scatter(
#                 x=results.index,
#                 y=results['final_signal'],
#                 name='Final Signal (with Regime)',
#                 line=dict(color='#2A9D8F', width=2)
#             ),
#             row=2, col=1
#         )
        
#         fig_zscore.update_xaxes(title_text="Date", row=2, col=1)
#         fig_zscore.update_yaxes(title_text="Z-Score", row=1, col=1)
#         fig_zscore.update_yaxes(title_text="Position", row=2, col=1)
        
#         fig_zscore.update_layout(
#             height=600,
#             hovermode='x unified',
#             template="plotly_white",
#             showlegend=True
#         )
        
#         st.plotly_chart(fig_zscore, use_container_width=True)
        
#         # Signal Statistics
#         st.markdown("---")
#         st.subheader(" Signal Statistics")
        
#         col1, col2, col3, col4 = st.columns(4)
        
#         long_signals = (results['final_signal'] == 1).sum()
#         short_signals = (results['final_signal'] == -1).sum()
#         total_signal_days = long_signals + short_signals
        
#         col1.metric("Long Position Days", long_signals)
#         col2.metric("Short Position Days", short_signals)
#         col3.metric("Neutral Days", len(results) - total_signal_days)
#         col4.metric("Active %", f"{(total_signal_days/len(results)*100):.1f}%")
        
#         # Entry/Exit Points
#         st.markdown("---")
#         st.subheader(" Entry & Exit Points")
        
#         # Detect entries and exits
#         position_change = results['final_signal'].diff()
#         entries = results[position_change != 0]
        
#         if len(entries) > 0:
#             fig_entries = go.Figure()
            
#             # Plot spread
#             fig_entries.add_trace(go.Scatter(
#                 x=results.index,
#                 y=results['spread_kf'],
#                 name='Spread',
#                 line=dict(color='lightgray', width=1),
#                 opacity=0.5
#             ))
            
#             # Mark entries
#             entry_long = entries[entries['final_signal'] == 1]
#             entry_short = entries[entries['final_signal'] == -1]
#             exit_points = entries[entries['final_signal'] == 0]
            
#             if len(entry_long) > 0:
#                 fig_entries.add_trace(go.Scatter(
#                     x=entry_long.index,
#                     y=entry_long['spread_kf'],
#                     mode='markers',
#                     name='Long Entry',
#                     marker=dict(color='green', size=12, symbol='triangle-up')
#                 ))
            
#             if len(entry_short) > 0:
#                 fig_entries.add_trace(go.Scatter(
#                     x=entry_short.index,
#                     y=entry_short['spread_kf'],
#                     mode='markers',
#                     name='Short Entry',
#                     marker=dict(color='red', size=12, symbol='triangle-down')
#                 ))
            
#             if len(exit_points) > 0:
#                 fig_entries.add_trace(go.Scatter(
#                     x=exit_points.index,
#                     y=exit_points['spread_kf'],
#                     mode='markers',
#                     name='Exit',
#                     marker=dict(color='blue', size=10, symbol='x')
#                 ))
            
#             fig_entries.update_layout(
#                 height=400,
#                 xaxis_title="Date",
#                 yaxis_title="Spread",
#                 template="plotly_white",
#                 hovermode='x unified'
#             )
            
#             st.plotly_chart(fig_entries, use_container_width=True)
            
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Long Entries", len(entry_long))
#             col2.metric("Short Entries", len(entry_short))
#             col3.metric("Exits", len(exit_points))
#         else:
#             st.info("No entry/exit points detected in the current backtest period")
    
#     # TAB 5: RISK ANALYSIS
#     with tab5:
#         st.header("Risk Analysis")
        
#         # Value at Risk
#         st.subheader(" Risk Metrics")
        
#         strategy_returns = results['strategy_return'].dropna()
        
#         # Calculate VaR and CVaR
#         var_95 = np.percentile(strategy_returns, 5)
#         var_99 = np.percentile(strategy_returns, 1)
#         cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        
#         col1, col2, col3, col4 = st.columns(4)
        
#         col1.metric("Daily VaR (95%)", f"{var_95*100:.2f}%")
#         col2.metric("Daily VaR (99%)", f"{var_99*100:.2f}%")
#         col3.metric("CVaR (95%)", f"{cvar_95*100:.2f}%")
#         col4.metric("Volatility", f"{strategy_returns.std()*np.sqrt(252)*100:.2f}%")
        
#         # Return Distribution
#         st.markdown("---")
#         st.subheader(" Return Distribution")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Histogram
#             fig_hist = go.Figure()
#             fig_hist.add_trace(go.Histogram(
#                 x=strategy_returns * 100,
#                 nbinsx=50,
#                 name='Returns',
#                 marker=dict(color='#2E86AB')
#             ))
            
#             fig_hist.add_vline(x=var_95*100, line_dash="dash", line_color="red", 
#                               annotation_text="VaR 95%")
#             fig_hist.add_vline(x=0, line_dash="solid", line_color="gray")
            
#             fig_hist.update_layout(
#                 height=350,
#                 xaxis_title="Daily Return (%)",
#                 yaxis_title="Frequency",
#                 template="plotly_white",
#                 showlegend=False
#             )
            
#             st.plotly_chart(fig_hist, use_container_width=True)
        
#         with col2:
#             # Q-Q Plot
#             from scipy import stats as sp_stats
            
#             theoretical_quantiles = sp_stats.norm.ppf(np.linspace(0.01, 0.99, 100))
#             sample_quantiles = np.percentile(strategy_returns, np.linspace(1, 99, 100))
            
#             fig_qq = go.Figure()
#             fig_qq.add_trace(go.Scatter(
#                 x=theoretical_quantiles,
#                 y=sample_quantiles,
#                 mode='markers',
#                 name='Q-Q Plot',
#                 marker=dict(color='#A23B72', size=6)
#             ))
            
#             # Add reference line
#             min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
#             max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
#             fig_qq.add_trace(go.Scatter(
#                 x=[min_val, max_val],
#                 y=[min_val, max_val],
#                 mode='lines',
#                 name='Normal',
#                 line=dict(color='red', dash='dash')
#             ))
            
#             fig_qq.update_layout(
#                 height=350,
#                 xaxis_title="Theoretical Quantiles",
#                 yaxis_title="Sample Quantiles",
#                 template="plotly_white"
#             )
            
#             st.plotly_chart(fig_qq, use_container_width=True)
        
#         # Regime Risk Analysis
#         st.markdown("---")
#         st.subheader("Risk by Market Regime")
        
#         safe_returns = results[results['regime'] == 1]['strategy_return'].dropna()
#         unsafe_returns = results[results['regime'] == 0]['math_only_return'].dropna()
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("**Safe Regime Performance**")
#             if len(safe_returns) > 0:
#                 st.metric("Mean Return", f"{safe_returns.mean()*252*100:.2f}% ann.")
#                 st.metric("Volatility", f"{safe_returns.std()*np.sqrt(252)*100:.2f}%")
#                 st.metric("Sharpe Ratio", f"{(safe_returns.mean()/safe_returns.std()*np.sqrt(252)):.2f}")
#                 st.metric("Win Rate", f"{(safe_returns > 0).mean()*100:.1f}%")
#             else:
#                 st.info("No data available")
        
#         with col2:
#             st.markdown("**Unsafe Regime (Avoided)**")
#             if len(unsafe_returns) > 0:
#                 st.metric("Mean Return", f"{unsafe_returns.mean()*252*100:.2f}% ann.")
#                 st.metric("Volatility", f"{unsafe_returns.std()*np.sqrt(252)*100:.2f}%")
#                 st.metric("Sharpe Ratio", f"{(unsafe_returns.mean()/unsafe_returns.std()*np.sqrt(252)):.2f}")
#                 st.metric("Win Rate", f"{(unsafe_returns > 0).mean()*100:.1f}%")
#             else:
#                 st.info("No data available")
        
#         # Rolling Volatility
#         st.markdown("---")
#         st.subheader(" Rolling Volatility (30-day)")
        
#         rolling_vol = strategy_returns.rolling(30).std() * np.sqrt(252) * 100
        
#         fig_vol = go.Figure()
#         fig_vol.add_trace(go.Scatter(
#             x=results.index[29:],
#             y=rolling_vol[29:],
#             fill='tozeroy',
#             line=dict(color='#F4A261', width=2),
#             fillcolor='rgba(244, 162, 97, 0.3)'
#         ))
        
#         fig_vol.update_layout(
#             height=300,
#             xaxis_title="Date",
#             yaxis_title="Annualized Volatility (%)",
#             template="plotly_white",
#             showlegend=False
#         )
        
#         st.plotly_chart(fig_vol, use_container_width=True)
    
#     # TAB 6: TRADE LOG
#     with tab6:
#         st.header("Trade Log & Analysis")
        
#         try:
#             trades = system.get_trade_analysis()
            
#             if len(trades) > 0:
#                 st.subheader("Trade Summary")
                
#                 col1, col2, col3, col4, col5 = st.columns(5)
                
#                 winning_trades = trades[trades['profitable']]
#                 losing_trades = trades[~trades['profitable']]
                
#                 col1.metric("Total Trades", len(trades))
#                 col2.metric("Winning Trades", len(winning_trades))
#                 col3.metric("Losing Trades", len(losing_trades))
#                 col4.metric("Win Rate", f"{(len(winning_trades)/len(trades)*100):.1f}%")
#                 col5.metric("Avg Duration", f"{trades['duration_days'].mean():.1f} days")
                
#                 st.markdown("---")
                
#                 # Trade Statistics
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader(" P&L Statistics")
#                     st.metric("Total P&L", f"{trades['pnl_pct'].sum():.2f}%")
#                     st.metric("Average Win", f"{winning_trades['pnl_pct'].mean():.2f}%" if len(winning_trades) > 0 else "N/A")
#                     st.metric("Average Loss", f"{losing_trades['pnl_pct'].mean():.2f}%" if len(losing_trades) > 0 else "N/A")
#                     st.metric("Best Trade", f"{trades['pnl_pct'].max():.2f}%")
#                     st.metric("Worst Trade", f"{trades['pnl_pct'].min():.2f}%")
                
#                 with col2:
#                     st.subheader("Trade Distribution")
                    
#                     fig_pnl_dist = go.Figure()
#                     fig_pnl_dist.add_trace(go.Histogram(
#                         x=trades['pnl_pct'],
#                         nbinsx=20,
#                         marker=dict(
#                             color=trades['pnl_pct'],
#                             colorscale='RdYlGn',
#                             cmin=-5,
#                             cmax=5
#                         )
#                     ))
                    
#                     fig_pnl_dist.update_layout(
#                         height=300,
#                         xaxis_title="P&L (%)",
#                         yaxis_title="Number of Trades",
#                         template="plotly_white",
#                         showlegend=False
#                     )
                    
#                     st.plotly_chart(fig_pnl_dist, use_container_width=True)
                
#                 # Trade Duration Analysis
#                 st.markdown("---")
#                 st.subheader("Trade Duration Analysis")
                
#                 fig_duration = go.Figure()
#                 fig_duration.add_trace(go.Scatter(
#                     x=trades['entry_date'],
#                     y=trades['duration_days'],
#                     mode='markers',
#                     marker=dict(
#                         size=abs(trades['pnl_pct']) * 5,
#                         color=trades['pnl_pct'],
#                         colorscale='RdYlGn',
#                         cmin=-5,
#                         cmax=5,
#                         showscale=True,
#                         colorbar=dict(title="P&L %")
#                     ),
#                     text=[f"P&L: {pnl:.2f}%<br>Duration: {dur} days" 
#                           for pnl, dur in zip(trades['pnl_pct'], trades['duration_days'])],
#                     hovertemplate='%{text}<extra></extra>'
#                 ))
                
#                 fig_duration.update_layout(
#                     height=350,
#                     xaxis_title="Entry Date",
#                     yaxis_title="Duration (Days)",
#                     template="plotly_white",
#                     showlegend=False
#                 )
                
#                 st.plotly_chart(fig_duration, use_container_width=True)
                
#                 # Detailed Trade Table
#                 st.markdown("---")
#                 st.subheader("Detailed Trade Log")
                
#                 # Format trades for display
#                 display_trades = trades.copy()
#                 display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
#                 display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
#                 display_trades['entry_signal'] = display_trades['entry_signal'].map({1: 'Long', -1: 'Short'})
#                 display_trades['profitable'] = display_trades['profitable'].map({True: '1', False: '0'})
                
#                 # Select columns to display
#                 display_cols = ['entry_date', 'exit_date', 'entry_signal', 'duration_days', 
#                                'entry_z', 'exit_z', 'pnl_pct', 'profitable']
                
#                 # Rename columns for better display
#                 display_trades = display_trades[display_cols].rename(columns={
#                     'entry_date': 'Entry Date',
#                     'exit_date': 'Exit Date',
#                     'entry_signal': 'Direction',
#                     'duration_days': 'Duration',
#                     'entry_z': 'Entry Z',
#                     'exit_z': 'Exit Z',
#                     'pnl_pct': 'P&L (%)',
#                     'profitable': 'Result'
#                 })
                
#                 # Style the dataframe
#                 def highlight_pnl(val):
#                     if isinstance(val, (int, float)):
#                         color = 'background-color: #d4edda' if val > 0 else 'background-color: #f8d7da'
#                         return color
#                     return ''
                
#                 st.dataframe(
#                     display_trades.style.applymap(highlight_pnl, subset=['P&L (%)']),
#                     use_container_width=True,
#                     height=400
#                 )
                
#                 # Download button
#                 csv = trades.to_csv(index=False)
#                 st.download_button(
#                     label=" Download Trade Log (CSV)",
#                     data=csv,
#                     file_name=f"trade_log_{ticker_y}_{ticker_x}.csv",
#                     mime="text/csv"
#                 )
                
#             else:
#                 st.info("No complete trades found in the backtest period. Try adjusting parameters or date range.")
                
#         except Exception as e:
#             st.error(f"Error analyzing trades: {str(e)}")
#             st.info("Trade analysis requires completed entry-exit cycles. Ensure your backtest has sufficient data.")

# else:
#     # Display welcome screen when no results
#     st.markdown("""
#     ## Enhanced Hybrid Pairs Trading System
    
#     dashboard provides comprehensive analysis of a sophisticated pairs trading strategy that combines:
    
#     ###  Key Features
    
#     1. **Adaptive Kalman Filter**
#        - Dynamic hedge ratio estimation
#        - Self-adjusting process noise (Q) based on market conditions
#        - Reduces persistent mis-hedging
    
#     2. **EWMA-Based Z-Score**
#        - More robust to outliers than traditional rolling statistics
#        - Better tracks recent regime changes
#        - Exponentially weighted mean and variance
    
#     3. **EMA Crossover Confirmation**
#        - Filters false signals from temporary spikes
#        - Reduces transaction costs
#        - Improves win rate and consistency
    
#     4. **ML Regime Detection**
#        - Random Forest classifier identifies safe/unsafe market conditions
#        - Avoids trading during high-volatility crisis periods
#        - Significantly improves risk-adjusted returns

#     ---
   
#     """)
   

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #888; padding: 2rem 0;'>
#     <p><b>Enhanced Hybrid Pairs Trading System</b></p>
#     <p style='font-size: 0.8rem;'> • Hardik Gupta • </p>
# </div>
# """, unsafe_allow_html=True)

# with pairs relation

# """
# app.py - Enhanced Streamlit Dashboard with Integrated Pair Finder
# Run with: streamlit run app.py
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime
# import sys
# import os
# # Ensure current directory is in path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# # Import your modules
# from hybrid_pairs_trading import (
#     HybridPairsTradingSystem,
#     download_data
# )

# try:
#     from pair_finder import PairFinder, get_precomputed_pairs
# except ImportError as e:
#     st.error(f"Could not import pair_finder: {e}")
#     st.error("Make sure pair_finder.py is in the same directory as app.py")
#     st.stop()

# # try:
# #     from pair_finder import PairFinder, get_precomputed_pairs
# # except ImportError:
# #     # Add the directory containing pair_finder.py to Python path
# #     sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# #     from pair_finder import PairFinder, get_precomputed_pairs

# # Page configuration
# st.set_page_config(
#     page_title="Hybrid Pairs Trading System",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .stButton>button {
#         width: 100%;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Title
# st.markdown('<p class="main-header">📈 Hybrid Pairs Trading System</p>', unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; color: #888;'>Kalman Filter + Machine Learning for Statistical Arbitrage</p>", unsafe_allow_html=True)
# st.markdown("---")

# # ============================================================================
# # SIDEBAR - CONFIGURATION
# # ============================================================================

# st.sidebar.header("⚙️ Configuration")

# # Market Selection
# market_type = st.sidebar.radio(
#     "🌍 Select Market",
#     options=['US Market', 'Indian Market (NSE)'],
#     index=0
# )

# market = 'US' if market_type == 'US Market' else 'INDIAN'

# # Tab selection in sidebar
# analysis_mode = st.sidebar.radio(
#     "📊 Analysis Mode",
#     options=['Quick Start (Precomputed Pairs)', 'Custom Pair', 'Find Best Pairs'],
#     index=0
# )

# # ============================================================================
# # MODE 1: QUICK START WITH PRECOMPUTED PAIRS
# # ============================================================================

# if analysis_mode == 'Quick Start (Precomputed Pairs)':
#     st.sidebar.subheader("📋 Select from Tested Pairs")
    
#     # Get precomputed pairs
#     precomputed = get_precomputed_pairs(market)
    
#     # Flatten pairs for selection
#     pair_options = []
#     for category, pairs in precomputed.items():
#         for pair in pairs:
#             label = f"{pair['y']} vs {pair['x']} - {pair['sector']} (Score: {pair['score']})"
#             pair_options.append({
#                 'label': label,
#                 'y': pair['y'],
#                 'x': pair['x'],
#                 'score': pair['score'],
#                 'category': category
#             })
    
#     # Sort by score
#     pair_options = sorted(pair_options, key=lambda x: x['score'], reverse=True)
    
#     selected_option = st.sidebar.selectbox(
#         "Choose pair:",
#         options=[p['label'] for p in pair_options],
#         index=0
#     )
    
#     # Get selected pair details
#     selected_pair = next(p for p in pair_options if p['label'] == selected_option)
#     ticker_y = selected_pair['y']
#     ticker_x = selected_pair['x']
    
#     # Show quality score
#     st.sidebar.success(f"✨ Quality Score: {selected_pair['score']}/100")
#     st.sidebar.info(f"📂 Category: {selected_pair['category']}")

# # ============================================================================
# # MODE 2: CUSTOM PAIR
# # ============================================================================

# elif analysis_mode == 'Custom Pair':
#     st.sidebar.subheader("🔧 Enter Custom Tickers")
    
#     ticker_y = st.sidebar.text_input(
#         "Stock Y (Dependent):",
#         value="HDFCBANK" if market == 'INDIAN' else "PEP"
#     )
    
#     ticker_x = st.sidebar.text_input(
#         "Stock X (Independent):",
#         value="ICICIBANK" if market == 'INDIAN' else "KO"
#     )
    
#     # Quality check button
#     if st.sidebar.button("🔍 Check Pair Quality"):
#         with st.spinner("Analyzing pair quality..."):
#             finder = PairFinder(market=market)
#             score, metrics = finder.calculate_pair_score(
#                 ticker_y, ticker_x,
#                 '2020-01-01', '2024-01-01'
#             )
            
#             st.session_state.pair_quality = {
#                 'score': score,
#                 'metrics': metrics
#             }
    
#     # Display quality if checked
#     if 'pair_quality' in st.session_state:
#         quality = st.session_state.pair_quality
#         score = quality['score']
        
#         if score >= 70:
#             st.sidebar.success(f"✅ Excellent Pair! Score: {score}/100")
#         elif score >= 50:
#             st.sidebar.warning(f"⚠️ Acceptable Pair. Score: {score}/100")
#         else:
#             st.sidebar.error(f"❌ Poor Pair. Score: {score}/100\nConsider choosing different stocks")

# # ============================================================================
# # MODE 3: FIND BEST PAIRS
# # ============================================================================

# else:  # Find Best Pairs
#     st.sidebar.subheader("🔍 Pair Discovery Settings")
    
#     finder = PairFinder(market=market)
#     universe = finder.get_stock_universe()
    
#     # Sector selection
#     sector_option = st.sidebar.selectbox(
#         "Select Sector",
#         options=['All Sectors'] + list(universe.keys())
#     )
    
#     sector = None if sector_option == 'All Sectors' else sector_option
    
#     min_score = st.sidebar.slider(
#         "Minimum Quality Score",
#         min_value=40,
#         max_value=90,
#         value=60,
#         step=5
#     )
    
#     max_pairs_to_test = st.sidebar.number_input(
#         "Max Pairs to Test",
#         min_value=10,
#         max_value=200,
#         value=50,
#         help="Limit computation time by testing fewer pairs"
#     )
    
#     if st.sidebar.button("🚀 Find Best Pairs"):
#         with st.spinner(f"Scanning {sector_option} for best pairs..."):
#             results = finder.find_best_pairs(
#                 sector=sector,
#                 min_score=min_score,
#                 start_date='2020-01-01',
#                 end_date='2024-01-01',
#                 max_pairs=max_pairs_to_test
#             )
            
#             st.session_state.pair_finder_results = results
    
#     # Display results if available
#     if 'pair_finder_results' in st.session_state:
#         results = st.session_state.pair_finder_results
        
#         if len(results) > 0:
#             st.sidebar.success(f"✅ Found {len(results)} good pairs!")
            
#             # Let user select from found pairs
#             pair_labels = [f"{row['Ticker_Y']} vs {row['Ticker_X']} ({row['Score']})" 
#                           for _, row in results.head(20).iterrows()]
            
#             selected_label = st.sidebar.selectbox("Choose pair:", pair_labels)
            
#             # Extract tickers
#             idx = pair_labels.index(selected_label)
#             ticker_y = results.iloc[idx]['Ticker_Y']
#             ticker_x = results.iloc[idx]['Ticker_X']
#         else:
#             st.sidebar.warning("No pairs found meeting criteria. Try lowering min score.")
#             ticker_y = "PEP"
#             ticker_x = "KO"
#     else:
#         # Default values
#         ticker_y = "HDFCBANK" if market == 'INDIAN' else "PEP"
#         ticker_x = "ICICIBANK" if market == 'INDIAN' else "KO"

# # ============================================================================
# # COMMON PARAMETERS
# # ============================================================================

# st.sidebar.markdown("---")
# st.sidebar.subheader("📅 Backtest Parameters")

# col1, col2 = st.sidebar.columns(2)
# start_date = col1.date_input("Start Date", datetime(2020, 1, 1))
# end_date = col2.date_input("End Date", datetime(2024, 1, 1))

# st.sidebar.subheader("🎯 Strategy Parameters")

# entry_z = st.sidebar.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
# exit_z = st.sidebar.slider("Exit Z-Score", 0.1, 1.0, 0.5, 0.1)
# train_period = st.sidebar.number_input("Training Period (days)", 100, 500, 252, 50)

# # Run Backtest Button
# st.sidebar.markdown("---")
# run_button = st.sidebar.button("🚀 Run Backtest", type="primary")

# # ============================================================================
# # INITIALIZE SESSION STATE
# # ============================================================================

# if 'results' not in st.session_state:
#     st.session_state.results = None
# if 'system' not in st.session_state:
#     st.session_state.system = None
# if 'metrics' not in st.session_state:
#     st.session_state.metrics = None

# # ============================================================================
# # RUN BACKTEST
# # ============================================================================

# if run_button:
#     with st.spinner(f"Running backtest for {ticker_y} vs {ticker_x}..."):
#         try:
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Download data
#             status_text.info(f"📥 Downloading data...")
#             progress_bar.progress(20)
            
#             # Handle Indian market tickers
#             if market == 'INDIAN':
#                 market_index_ticker = '^NSEI'  # Nifty 50
#                 download_y = f"{ticker_y}.NS"
#                 download_x = f"{ticker_x}.NS"
#             else:
#                 market_index_ticker = 'SPY'
#                 download_y = ticker_y
#                 download_x = ticker_x
            
#             stock_y, stock_x, market_index = download_data(
#                 download_y,
#                 download_x,
#                 market_index_ticker,
#                 start_date.strftime('%Y-%m-%d'),
#                 end_date.strftime('%Y-%m-%d')
#             )
            
#             progress_bar.progress(40)
            
#             # Initialize system
#             status_text.info("🔧 Initializing system...")
#             system = HybridPairsTradingSystem()
#             progress_bar.progress(50)
            
#             # Run backtest
#             status_text.info("🔄 Running backtest...")
#             results = system.run_backtest(
#                 stock_y=stock_y,
#                 stock_x=stock_x,
#                 market_index=market_index,
#                 train_period=train_period,
#                 entry_z=entry_z,
#                 exit_z=exit_z
#             )
#             progress_bar.progress(80)
            
#             # Get metrics
#             status_text.info("📊 Calculating metrics...")
#             metrics = system.get_performance_metrics()
#             progress_bar.progress(100)
            
#             # Store in session state
#             st.session_state.results = results
#             st.session_state.system = system
#             st.session_state.metrics = metrics
#             st.session_state.ticker_y = ticker_y
#             st.session_state.ticker_x = ticker_x
#             st.session_state.market = market
            
#             status_text.success("✅ Backtest completed!")
#             progress_bar.empty()
#             status_text.empty()
            
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")
#             st.exception(e)
#             st.stop()

# # ============================================================================
# # DISPLAY RESULTS
# # ============================================================================

# if st.session_state.results is not None:
#     results = st.session_state.results
#     system = st.session_state.system
#     metrics = st.session_state.metrics
#     ticker_y = st.session_state.ticker_y
#     ticker_x = st.session_state.ticker_x
#     market = st.session_state.market
    
#     # Create tabs
#     tab1, tab2, tab3, tab4 = st.tabs([
#         "📊 Overview",
#         "📈 Performance",
#         "🎯 Signals",
#         "📋 Trades"
#     ])
    
#     # TAB 1: OVERVIEW
#     with tab1:
#         st.header(f"Performance Overview: {ticker_y} vs {ticker_x}")
        
#         hybrid_metrics = metrics['Hybrid Strategy']
#         math_metrics = metrics['Math Only']
        
#         # Key Metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("💰 Total Return (Hybrid)", hybrid_metrics['Total Return'])
        
#         with col2:
#             st.metric("📊 Sharpe Ratio", hybrid_metrics['Sharpe Ratio'])
        
#         with col3:
#             st.metric("📉 Max Drawdown", hybrid_metrics['Max Drawdown'])
        
#         with col4:
#             st.metric("🎯 Win Rate", hybrid_metrics['Win Rate'])
        
#         st.markdown("---")
        
#         # Cumulative Returns Chart
#         st.subheader("📈 Cumulative Returns")
        
#         fig = go.Figure()
        
#         fig.add_trace(go.Scatter(
#             x=results.index,
#             y=(results['strategy_cumulative'] - 1) * 100,
#             name='Hybrid',
#             line=dict(color='#2A9D8F', width=3)
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=results.index,
#             y=(results['math_only_cumulative'] - 1) * 100,
#             name='Math Only',
#             line=dict(color='#E76F51', width=2, dash='dash')
#         ))
        
#         fig.update_layout(
#             height=400,
#             hovermode='x unified',
#             template="plotly_white",
#             yaxis_title="Return (%)"
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Side-by-side comparison
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("🔵 Hybrid Strategy")
#             for key, value in hybrid_metrics.items():
#                 st.metric(key, value)
        
#         with col2:
#             st.subheader("🔴 Math Only")
#             for key, value in math_metrics.items():
#                 st.metric(key, value)
    
#     # TAB 2: PERFORMANCE
#     with tab2:
#         st.header("Detailed Performance Analysis")
        
#         # Prices
#         st.subheader("Stock Prices")
#         fig = go.Figure()
        
#         y_norm = (results['Y'] / results['Y'].iloc[0]) * 100
#         x_norm = (results['X'] / results['X'].iloc[0]) * 100
        
#         fig.add_trace(go.Scatter(x=results.index, y=y_norm, name=ticker_y))
#         fig.add_trace(go.Scatter(x=results.index, y=x_norm, name=ticker_x))
        
#         fig.update_layout(height=400, template="plotly_white")
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Drawdown
#         st.subheader("Drawdown Analysis")
        
#         hybrid_cum = results['strategy_cumulative']
#         hybrid_dd = ((hybrid_cum - hybrid_cum.expanding().max()) / hybrid_cum.expanding().max()) * 100
        
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=results.index,
#             y=hybrid_dd,
#             fill='tozeroy',
#             name='Drawdown'
#         ))
        
#         fig.update_layout(height=300, template="plotly_white")
#         st.plotly_chart(fig, use_container_width=True)
    
#     # TAB 3: SIGNALS
#     with tab3:
#         st.header("Trading Signals")
        
#         fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4])
        
#         # Z-Score
#         fig.add_trace(go.Scatter(
#             x=results.index,
#             y=results['z_score'],
#             name='Z-Score'
#         ), row=1, col=1)
        
#         fig.add_hline(y=entry_z, line_dash="dash", line_color="red", row=1, col=1)
#         fig.add_hline(y=-entry_z, line_dash="dash", line_color="red", row=1, col=1)
        
#         # Signals
#         fig.add_trace(go.Scatter(
#             x=results.index,
#             y=results['final_signal'],
#             name='Signal',
#             fill='tozeroy'
#         ), row=2, col=1)
        
#         fig.update_layout(height=600, template="plotly_white")
#         st.plotly_chart(fig, use_container_width=True)
    
#     # TAB 4: TRADES
#     with tab4:
#         st.header("Trade Log")
        
#         trades = system.get_trade_analysis()
        
#         if len(trades) > 0:
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.metric("Total Trades", len(trades))
#             with col2:
#                 st.metric("Win Rate", f"{trades['profitable'].mean()*100:.1f}%")
#             with col3:
#                 st.metric("Avg P&L", f"{trades['pnl_pct'].mean():.2f}%")
            
#             st.dataframe(trades, use_container_width=True, height=400)
#         else:
#             st.warning("No trades found")

# # ============================================================================
# # WELCOME SCREEN
# # ============================================================================

# else:
#     st.info("👈 Configure parameters and click **Run Backtest**")
    
#     # Show pair finder results if available
#     if 'pair_finder_results' in st.session_state:
#         st.header("🔍 Discovered Pairs")
#         results = st.session_state.pair_finder_results
        
#         if len(results) > 0:
#             st.dataframe(results, use_container_width=True, height=400)
            
#             st.download_button(
#                 "📥 Download Results CSV",
#                 data=results.to_csv(index=False),
#                 file_name=f"pairs_{market}_{datetime.now().strftime('%Y%m%d')}.csv",
#                 mime="text/csv"
#             )
    
#     # System overview
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         ### 📐 Mathematical Core
#         - Kalman Filter for dynamic hedge ratio
#         - Mean reversion detection
#         - Z-score based signals
#         """)
    
#     with col2:
#         st.markdown("""
#         ### 🤖 ML Component
#         - Random Forest classifier
#         - Regime detection (SAFE/UNSAFE)
#         - Risk management filter
#         """)

# # Footer
# st.markdown("---")
# st.markdown("<p style='text-align: center; color: gray;'>Hybrid Pairs Trading System | Educational Purposes Only</p>", unsafe_allow_html=True)