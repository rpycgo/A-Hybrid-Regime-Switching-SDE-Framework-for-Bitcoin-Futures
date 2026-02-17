import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from dateutil.relativedelta import relativedelta
from arch import arch_model

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.utils.configuration_loader import ConfigurationLoader
from src.data.preprocessor import Preprocessor
from src.data.dataset_builder import DatasetBuilder
from src.engines.backtester import BacktestEngine
from src.utils.performance_analyzer import PerformanceAnalyzer
from src.visualization.plotter import StrategyPlotter


def process_single_window_garch(test_start_date, train_data, full_data, engine):
    """
    Executes an enhanced GARCH(1,1) benchmark with asymmetric alpha estimation
    and optimized volatility forecasting.
    """
    # 1. Window Slicing
    training_end = test_start_date - pd.Timedelta(seconds=1)
    training_start = training_end - relativedelta(months=3)
    testing_end = test_start_date + relativedelta(months=1) - pd.Timedelta(seconds=1)

    if testing_end > pd.to_datetime('2026-01-31'):
        testing_end = pd.to_datetime('2026-01-31')

    # 2. Slice Data
    training_slice = train_data.loc[training_start:training_end].copy()
    testing_slice = full_data.loc[test_start_date:testing_end].copy()

    if len(training_slice) < 100:
        return None

    # 3. GARCH(1,1) Estimation with Robust Scaling
    # Use rescale=True to let the optimizer find the best scale for convergence
    train_returns = training_slice['log_return'] * 100
    model = arch_model(train_returns, vol='Garch', p=1, q=1, dist='normal', rescale=True)
    res = model.fit(disp='off', show_warning=False)

    # 4. Optimized Volatility Forecasting (Reflecting Grok's Feedback)
    # Using the built-in forecast() method for better stability than manual loops
    horizon = len(testing_slice)
    forecasts = res.forecast(horizon=horizon, reindex=False)

    # Extract variance and convert to volatility.
    # Must divide by res.scale to bring it back to the original log_return scale.
    internal_scale = res.scale if hasattr(res, 'scale') else 1.0
    forecast_vol = np.sqrt(forecasts.variance.values.flatten()) / (internal_scale * 100)

    testing_slice['GARCH_Vol'] = forecast_vol

    # 5. Signal Generation with Fixed SDE Baseline Parameters
    k_avg = 0.45
    gamma_avg = 2.70

    # Calculate Z-score using predicted GARCH volatility
    z_score = (
        (testing_slice['Close'] - testing_slice['Open']) /
        (testing_slice['Open'] * testing_slice['GARCH_Vol'].replace(0, np.nan))
    )
    testing_slice['regime_prob'] = 1 / (1 + np.exp(-k_avg * (z_score.abs() - gamma_avg)))

    # Apply Filters (Sticky, ADX) via Engine
    testing_with_signals = engine.generate_regime_signals(
        testing_slice,
        k_estimated=k_avg,
        gamma_estimated=gamma_avg,
    )

    # 6. Asymmetric Parameter Scaling (Reflecting Grok's Feedback)
    # Separating long and short alphas to account for crypto market bias
    pos_ret = training_slice[training_slice['log_return'] > 0]['log_return']
    neg_ret = training_slice[training_slice['log_return'] < 0]['log_return']

    estimates = {
        'alpha_long': pos_ret.mean() * 100 if not pos_ret.empty else 0.5,
        'alpha_short': abs(neg_ret.mean()) * 100 if not neg_ret.empty else 0.5,
        # Use the last estimated conditional volatility as the sigma_1 proxy
        'sigma_1': (res.conditional_volatility.iloc[-1] / internal_scale) / 100
    }

    # Dynamic adjustment of TP/SL/Trailing parameters based on SNR
    base_config = engine.trading_parameters.copy()
    dynamic_params = engine.adjust_parameters_by_snr(estimates, base_config)

    # 7. Run Simulation
    trades = engine.run_backtest_simulation(testing_with_signals, dynamic_params)

    return {
        'window': test_start_date.strftime('%Y-%m-%d'),
        'trades': trades,
        'summary': res.params.to_frame().T,
        'signal_df': testing_with_signals[['Close', 'regime_prob']]
    }


def run_garch_benchmark_pipeline():
    """
    Main pipeline to run GARCH Walk-Forward Analysis.
    Synced with ConfigurationLoader and DatasetBuilder.
    """
    config_loader = ConfigurationLoader()
    raw_data = pd.read_csv('./data/btcusdt_future.csv', index_col=0, parse_dates=True)

    preprocessor = Preprocessor(config_loader)
    builder = DatasetBuilder(config_loader)

    # Data Preparation (Identical to SDE)
    data = preprocessor.calculate_base_features(raw_data)
    data = preprocessor.identify_discovery_sr_levels(data)
    events = builder.load_events()
    data = builder.apply_event_tagging(data, events)
    data = preprocessor.calculate_directional_indicators(data)
    data = preprocessor.calculate_strategy_indicators(data)

    train_data = builder.slice_training_data(data)
    full_data = data 

    engine = BacktestEngine(config_loader)
    wfa_config = config_loader.get_backtest_settings()['walk_forward_settings']

    test_start_dates = pd.date_range(start=wfa_config['start_date'],
                                     end=wfa_config['end_date'],
                                     freq='MS')

    print(f"🚀 Starting Parallel GARCH Benchmark: {len(test_start_dates)} windows...")

    results = Parallel(n_jobs=wfa_config['parallel_jobs'])(
        delayed(process_single_window_garch)(dt, train_data, full_data, engine)
        for dt in test_start_dates
    )

    # Consolidation
    all_trades, all_signals, all_summaries = [], [], []
    for result in results:
        if result is None:
            continue

        all_trades.append(result['trades'])
        all_signals.append(result['signal_df'])
        all_summaries.append(result['summary'])

    final_trades = pd.concat(all_trades).sort_values(by=['entry_time']).reset_index(drop=True)
    final_signals = pd.concat(all_signals).sort_index()

    # Results save
    final_trades = (
        pd.concat(all_trades)
        .sort_values(by=['entry_time'])
        .reset_index(drop=True)
    )
    final_trades.to_csv("results/garch_trade_results.csv")

    # Reporting
    print(f"\n🏆 [Benchmark] GARCH(1,1) Results")
    metrics, equity_curve, drawdown = PerformanceAnalyzer.calculate_metrics(final_trades)
    PerformanceAnalyzer.print_performance_report(metrics)
    PerformanceAnalyzer.calculate_predictive_ic(final_signals, 'regime_prob')

    # Visualization
    save_path = (
    f'figures/{base_name}'
    if (base_name := os.path.splitext(os.path.basename(__file__))[0]) != 'run_backtester'
    else 'figures/sde'
    )

    plotter = StrategyPlotter()
    plotter.plot_backtest_results(
        equity_curve,
        drawdown,
        save_path,
    )


if __name__ == "__main__":
    run_garch_benchmark_pipeline()
