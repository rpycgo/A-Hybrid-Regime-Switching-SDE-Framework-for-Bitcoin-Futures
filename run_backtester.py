import pandas as pd
import pickle
from joblib import Parallel, delayed
from dateutil.relativedelta import relativedelta

from src.utils.configuration_loader import ConfigurationLoader
from src.data.preprocessor import Preprocessor
from src.data.dataset_builder import DatasetBuilder
from src.models.sde_model import SdeModeler
from src.engines.backtester import BacktestEngine
from src.utils.performance_analyzer import PerformanceAnalyzer
from src.visualization.plotter import StrategyPlotter


def process_single_window_task(test_start_date, train_data, full_data, modeler, engine):
    """Executes training and testing for one Walk-Forward window."""
    # 1. Window Slicing
    training_end = test_start_date - pd.Timedelta(seconds=1)
    training_start = training_end - relativedelta(months=3)
    testing_end = test_start_date + relativedelta(months=1) - pd.Timedelta(seconds=1)

    # Research End Boundary
    if testing_end > pd.to_datetime('2026-01-31'):
        testing_end = pd.to_datetime('2026-01-31')

    # 2. Slice Data (Training from in-zone data, Testing from full price action)
    training_slice = train_data.loc[training_start:training_end].copy()
    testing_slice = full_data.loc[test_start_date:testing_end].copy()

    # 3. Estimate SDE Parameters
    trace, summary_df, estimates = modeler.estimate_parameters(
        z_values=training_slice['hybrid_z_score'].values,
        returns_scaled=training_slice['log_return'].values * 100,
        direction=training_slice['direction_indicator'].values
    )

    if estimates is None:
        return pd.DataFrame()

    # 4. Signal Generation and Parameter Scaling
    testing_with_signals = engine.generate_regime_signals(testing_slice,
                                                          estimates['k'],
                                                          estimates['gamma'])

    base_config = {
        'tp_long': engine.trading_parameters['tp_long'],
        'sl_long': engine.trading_parameters['sl_long'],
        'tp_short': engine.trading_parameters['tp_short'],
        'sl_short': engine.trading_parameters['sl_short'],
        'trailing_stop_start_ratio': engine.trading_parameters['trailing_stop_start_ratio'],
        'max_hold_hours': engine.trading_parameters['max_hold_hours']
    }
    dynamic_params = engine.adjust_parameters_by_snr(estimates, base_config)

    # 5. Run Simulation
    trades = engine.run_backtest_simulation(testing_with_signals, dynamic_params)

    return {
        'window': test_start_date.strftime('%Y-%m-%d'),
        'trades': trades,
        'trace': trace,
        'summary': summary_df,
        'signal_df': testing_with_signals[['Close', 'regime_prob']],
    }


def run_walk_forward_analysis_pipeline():
    # --- Data Loading & Preprocessing ---
    config_loader = ConfigurationLoader()
    raw_data = pd.read_csv('./data/btcusdt_future.csv', index_col=0, parse_dates=True)

    # Using Pre-existing Processors
    preprocessor = Preprocessor(config_loader)
    builder = DatasetBuilder(config_loader)

    # Full Data Preparation
    data = preprocessor.calculate_base_features(raw_data)
    data = preprocessor.identify_discovery_sr_levels(data)
    events = builder.load_events()
    data = builder.apply_event_tagging(data, events)
    data = preprocessor.calculate_directional_indicators(data)
    data = preprocessor.calculate_strategy_indicators(data)

    # Split for Walk-Forward Analysis
    train_data = builder.slice_training_data(data)
    full_data = data # Testing uses the full featured dataset

    # --- Setup Engines ---
    engine = BacktestEngine(config_loader)
    model = SdeModeler(config_loader)
    wfa_config = config_loader.get_backtest_settings()['walk_forward_settings']

    test_start_dates = pd.date_range(start=wfa_config['start_date'],
                                     end=wfa_config['end_date'],
                                     freq='MS')

    print(f"🚀 Initializing Parallel Walk-Forward Analysis: {len(test_start_dates)} windows...")

    results = Parallel(n_jobs=wfa_config['parallel_jobs'])(
        delayed(process_single_window_task)(dt, train_data, full_data, model, engine)
        for dt in test_start_dates
    )

    # --- Consolidation ---
    all_trades = []
    all_summaries = {}
    all_traces = {}
    all_signals = []

    for result in results:
        if result is None:
            continue

        window = result['window']
        all_trades.append(result['trades'])
        all_signals.append(result['signal_df'])

        summary = result['summary'].copy()
        summary['window'] = window
        all_summaries[window] = summary

        all_traces[window] = result['trace']

    # Results save
    final_trades = (
        pd.concat(all_trades)
        .sort_values(by=['entry_time'])
        .reset_index(drop=True)
    )
    final_trades.to_csv("results/results.csv")

    final_signals = pd.concat(all_signals).sort_index()

    final_summary_df = pd.concat(all_summaries.values())
    final_summary_df.to_csv("results/all_windows_summary.csv")

    with open("results/all_windows_traces.pkl", "wb") as file:
        pickle.dump(all_traces, file)

    print(f"✅ Walk-Forward Analysis Complete. {len(all_traces)} windows saved in one go.")

    # Report
    # 1. Performance Metrics Report
    metrics, equity_curve, drawdown = PerformanceAnalyzer.calculate_metrics(final_trades)
    PerformanceAnalyzer.print_performance_report(metrics)

    # 2. Predictive Power Report (IC Analysis)
    PerformanceAnalyzer.calculate_predictive_ic(final_signals, 'regime_prob')

    # 3. Visualization
    plotter = StrategyPlotter()
    plotter.plot_backtest_results(equity_curve, drawdown)
    print(f"📊 Figures and Report generated successfully.")


if __name__ == "__main__":
    run_walk_forward_analysis_pipeline()
