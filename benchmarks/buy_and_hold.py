import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to sys.path to enable imports from the 'src' directory
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils.configuration_loader import ConfigurationLoader


def run_buy_and_hold_benchmark():
    """
    Executes a standalone Buy-and-Hold benchmark analysis.
    This script calculates academic-standard metrics (Sharpe, Sortino, T-stat, MDD)
    based on the daily equity curve to ensure a fair comparison with the SDE model.
    """
    # 1. Initialization and Data Loading
    config_loader = ConfigurationLoader()
    data_path = root_path / "data" / "btcusdt_future.csv"

    if not data_path.exists():
        print(f"❌ Data file not found at: {data_path}")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Define the research period (Match with the LaTeX table: 2024-2026)
    wfa_settings = config_loader.get_backtest_settings()['walk_forward_settings']
    test_start = wfa_settings['start_date']
    test_end = wfa_settings['end_date']

    df = df.loc[test_start:test_end].copy()

    if df.empty:
        print(f"❌ No data available for the period: {test_start} to {test_end}")
        return

    # 2. Daily Equity Curve and Return Calculation
    # Resampling to daily frequency is essential for capturing volatility (Risk)
    daily_price = df['Close'].resample('D').last().ffill()
    daily_returns = daily_price.pct_change().dropna()

    # Calculate total duration in days for accurate annualization
    total_days = (daily_price.index[-1] - daily_price.index[0]).days
    if total_days <= 0:
        total_days = 1 # Avoid division by zero

    # 3. Core Performance Metrics Calculation
    # Total Return: (Ending Price / Starting Price) - 1
    total_return = (daily_price.iloc[-1] / daily_price.iloc[0]) - 1

    # Annualized Return: Compound growth rate over a 365-day basis
    annual_return = (1 + total_return) ** (365 / total_days) - 1

    # Sharpe Ratio: Annualized excess return per unit of total risk (Standard Deviation)
    # Risk-free rate (rf) is assumed to be 0 for benchmark consistency
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)

    # Sortino Ratio: Annualized excess return per unit of downside risk
    downside_returns = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(365)

    # Maximum Drawdown (MDD): The largest peak-to-trough decline in equity
    rolling_max = daily_price.cummax()
    drawdown = (daily_price - rolling_max) / rolling_max
    mdd = drawdown.min()

    # T-statistic: Tests if the mean return is significantly different from zero
    t_stat = daily_returns.mean() / (daily_returns.std() / np.sqrt(len(daily_returns)))

    # Average Recovery Time
    is_in_drawdown = drawdown < 0
    drawdown_groups = (is_in_drawdown != is_in_drawdown.shift()).cumsum()
    recovery_periods = is_in_drawdown.groupby(drawdown_groups).sum()
    recovery_durations = recovery_periods[recovery_periods > 0]
    avg_recovery_time = recovery_durations.mean() if not recovery_durations.empty else 0

    # 4. Final Performance Report
    print("=" * 60)
    print(f"       🏆 BENCHMARK REPORT: BUY-AND-HOLD")
    print(f"       Analysis Period: {test_start} - {test_end}")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value'}")
    print("-" * 60)
    print(f"{'Total Executed Trades':<30} | {len(daily_returns):>13}")
    print(f"{'Total Return (%)':<30} | {total_return * 100:>12.2f}%")
    print(f"{'Annualized Return (%)':<30} | {annual_return * 100:>12.2f}%")
    print(f"{'Max Drawdown (%)':<30} | {mdd * 100:>12.2f}%")
    print(f"{'Sharpe Ratio':<30} | {sharpe:>13.2f}")
    print(f"{'Sortino Ratio':<30} | {sortino:>13.2f}")
    print(f"{'Avg. Recovery Time':<30} | {avg_recovery_time:>10.2f} Days")
    print(f"{'T-statistic':<30} | {t_stat:>13.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_buy_and_hold_benchmark()
