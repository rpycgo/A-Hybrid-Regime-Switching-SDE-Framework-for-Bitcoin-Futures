import sys
import os
import argparse
import pandas as pd
import pickle

from src.visualization.plotter import StrategyPlotter


def main():
    """
    Main entry point for SDE backtest result analysis.
    Decouples simulation from visualization for efficiency.
    """
    parser = argparse.ArgumentParser(description="Advanced SDE Strategy Results Analyzer")

    # Path Configuration
    parser.add_argument('--summary', type=str, default='results/all_windows_summary.csv', 
                        help='Path to the aggregated windows summary CSV')
    parser.add_argument('--traces', type=str, default='results/all_windows_traces.pkl', 
                        help='Path to the pickled MCMC traces dictionary')
    parser.add_argument('--out', type=str, default='figures', 
                        help='Directory where generated figures will be stored')

    # Analysis Scopes
    parser.add_argument('--params', nargs='+', 
                        default=['alpha_long', 'alpha_short', 'gamma',
                                 'kappa', 'k', 'sigma_0', 'sigma_1'],
                        help='List of parameters for Drift and Ridge plots')

    # Diagnostics Selection
    parser.add_argument('--window', type=str, 
                        help='Target window date (YYYY-MM-DD) for specific MCMC diagnostics')
    parser.add_argument('--diag_all', action='store_true', 
                        help='Flag to run full diagnostics for every available window')

    args = parser.parse_args()

    # Initialize Plotter
    plotter = StrategyPlotter()
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # --- Section 1: Evolutionary Dynamics (Drift & Ridge Plots) ---
    if os.path.exists(args.summary):
        print(f"📈 Processing Evolutionary Dynamics from {args.summary}...")
        summary_df = pd.read_csv(args.summary, index_col=0)

        for param in args.params:
            print(f"   > Generating Drift and Ridge plots for: {param}")
            plotter.draw_drift_plot(summary_df, param, args.out)
            plotter.draw_ridge_plot(summary_df, param, args.out)
    else:
        print(f"⚠️ Warning: Summary file not found at {args.summary}. Skipping Drift/Ridge plots.")

    # --- Section 2: MCMC Diagnostics (Trace, Pair, Rank Plots) ---
    if args.window or args.diag_all:
        if os.path.exists(args.traces):
            print(f"🔍 Loading MCMC traces from {args.traces}...")
            try:
                with open(args.traces, 'rb') as f:
                    combined_traces = pickle.load(f)
            except Exception as e:
                print(f"❌ Error loading pickle: {e}")
                sys.exit(1)

            # Single Window Diagnostics
            if args.window:
                if args.window in combined_traces:
                    plotter.plot_window_diagnostics(combined_traces[args.window],
                                                    args.window,
                                                    args.out)
                else:
                    print(f"❌ Window '{args.window}' not found in trace file.")
                    print(f"   Available windows: {list(combined_traces.keys())}")

            # Batch Diagnostics for All Windows
            if args.diag_all:
                print(f"🚀 Running batch diagnostics for all {len(combined_traces)} windows...")
                for window_name, trace in combined_traces.items():
                    plotter.plot_window_diagnostics(trace, window_name, args.out)
        else:
            print(f"⚠️ Warning: Traces file not found at {args.traces}. Skipping diagnostics.")

    print(f"\n✅ Analysis complete. All assets are saved in the '{args.out}/' directory.")


if __name__ == "__main__":
    main()
