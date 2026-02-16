import pandas as pd
import numpy as np
from scipy import stats


class PerformanceAnalyzer:
    """
    A specialized engine for calculating quantitative trading metrics.
    Decouples mathematical analysis from visualization.
    """
    @staticmethod
    def calculate_metrics(results_df: pd.DataFrame):
        """
        Computes comprehensive performance statistics from trade-level results.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing 'PnL' and 'exit_time'.
            
        Returns:
            tuple: (metrics_dict, equity_curve, drawdown_series)
        """
        if results_df is None or results_df.empty:
            return None, None, None

        # Ensure datetime conversion
        results_df['exit_time'] = pd.to_datetime(results_df['exit_time'])
        results_df = results_df.sort_values('exit_time')

        # --- 1. Series Calculation ---
        equity_curve = (1 + results_df['PnL']).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max

        # --- 2. Basic Performance Metrics ---
        total_trades = len(results_df)
        total_return = (equity_curve.iloc[-1] - 1) * 100

        # Duration for annualization
        duration_days = (results_df['exit_time'].max() - results_df['exit_time'].min()).days
        duration_days = max(duration_days, 1)  # Prevent division by zero

        # Annualized Return (CAGR approximation)
        annualized_return = ((1 + total_return/100) ** (365 / duration_days) - 1) * 100

        # --- 3. Drawdown & Recovery Time ---
        max_drawdown = drawdown.min() * 100
        avg_recovery_days = PerformanceAnalyzer._calculate_recovery_time(drawdown, duration_days, total_trades)

        # --- 4. Risk-Adjusted Returns (Sharpe & Sortino) ---
        mean_pnl = results_df['PnL'].mean()
        std_pnl = results_df['PnL'].std()

        # Trade-based annualization factor
        ann_factor = np.sqrt(total_trades / (duration_days / 365))

        # Sharpe Ratio: Mean Return / Standard Deviation
        sharpe_ratio = (mean_pnl / std_pnl) * ann_factor if std_pnl != 0 else 0

        # Sortino Ratio: Mean Return / Downside Deviation
        downside_pnl = results_df[results_df['PnL'] < 0]['PnL']
        downside_std = downside_pnl.std()
        sortino_ratio = (mean_pnl / downside_std) * ann_factor if downside_std != 0 else 0

        # --- 5. Statistical Significance (T-test) ---
        t_stat, p_value = stats.ttest_1samp(results_df['PnL'], 0)

        # --- 6. Win Rate & Trade Distribution ---
        win_rate = (len(results_df[results_df['PnL'] > 0]) / total_trades) * 100
        avg_pnl = results_df['PnL'].mean() * 100

        metrics = {
            "total_trades": total_trades,
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "t_stat": t_stat,
            "p_value": p_value,
            "avg_recovery_days": avg_recovery_days,
            "win_rate_pct": win_rate,
            "avg_pnl_pct": avg_pnl
        }

        return metrics, equity_curve, drawdown

    @staticmethod
    def _calculate_recovery_time(drawdown, duration_days, total_trades):
        """Internal helper to estimate average recovery time from drawdown periods."""
        is_in_drawdown = drawdown < 0
        recovery_times = []
        current_recovery = 0

        for in_dd in is_in_drawdown:
            if in_dd:
                current_recovery += 1
            else:
                if current_recovery > 0:
                    recovery_times.append(current_recovery)
                current_recovery = 0

        # Convert trade counts back to approximate calendar days
        if recovery_times:
            return np.mean(recovery_times) * (duration_days / total_trades)
        return 0

    @staticmethod
    def print_performance_report(metrics):
        """
        Prints a standalone, formatted ASCII performance report.
        Designed for quick CLI feedback after a backtest.
        """
        if not metrics:
            print("⚠️ No metrics available to report.")
            return

        # Header
        print("\n" + "="*80)
        print(f"{'📊 STRATEGY PERFORMANCE REPORT':^80}")
        print("="*80)

        # Human-readable labels for the metrics dictionary
        label_map = [
            ("total_trades", "Total Executed Trades"),
            ("total_return_pct", "Total Return (%)"),
            ("annualized_return_pct", "Annualized Return (%)"),
            ("max_drawdown_pct", "Max Drawdown (%)"),
            ("sharpe_ratio", "Sharpe Ratio"),
            ("sortino_ratio", "Sortino Ratio"),
            ("win_rate_pct", "Win Rate (%)"),
            ("avg_pnl_pct", "Avg. PnL per Trade (%)"),
            ("avg_recovery_days", "Avg. Recovery Time"),
            ("t_stat", "T-statistic")
        ]

        for key, label in label_map:
            if key in metrics:
                val = metrics[key]
                
                # Special handling for T-statistic significance
                if key == "t_stat":
                    p_val = metrics.get("p_value", 1.0)
                    stars = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"{label:<30} : {val:.2f}{stars}")
                
                # Generic float formatting
                elif isinstance(val, (float, np.float64)):
                    print(f"{label:<30} : {val:.2f}")
                
                # Integer formatting
                else:
                    print(f"{label:<30} : {val}")

        # Footer with Legend
        print("="*80)
        print(f"(* p < 0.05, ** p < 0.01) | SDE Regime Strategy Analysis")
        print("="*80 + "\n")
