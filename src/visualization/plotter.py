import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import arviz as az
import matplotlib.pyplot as plt


class StrategyPlotter:
    """
    A visualization engine for analyzing SDE model parameters and MCMC diagnostics.
    Supports evolutionary dynamics (Drift/Ridge) and convergence checks (Trace/Pair/Rank).
    """
    def __init__(self):
        """
        Initializes the plotter with high-resolution publication settings.
        """
        # Global visualization configuration
        plt.rcParams.update({
            'font.size': 18,
            'axes.labelsize': 20,
            'axes.titlesize': 22,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 16,
            'figure.dpi': 300
        })

        # Target SDE parameters for analysis
        self.params = ['alpha_long', 'alpha_short', 'gamma', 'kappa', 'k', 'sigma_0', 'sigma_1']

        # Mapping for LaTeX mathematical notation
        self.greek_map = {
            'alpha_long': r'\alpha_{long}',
            'alpha_short': r'\alpha_{short}',
            'gamma': r'\gamma',
            'kappa': r'\kappa',
            'k': r'k',
            'sigma_0': r'\sigma_0',
            'sigma_1': r'\sigma_1'
        }

    def _get_latex_label(self, param_name):
        """Returns formatted LaTeX string for plot labels."""
        label = self.greek_map.get(param_name, param_name)
        return f"${{{label}}}$"

    def draw_drift_plot(self, df, target_param, save_path, interval=2):
        """
        Plots the time-series evolution of a parameter's posterior mean and 95% HDI.
        """
        os.makedirs(save_path, exist_ok=True)

        subset = df[df.index == target_param].copy()
        subset['date_date'] = pd.to_datetime(subset['window'])
        subset = subset.sort_values('date_date')

        latex_name = self._get_latex_label(target_param)

        plt.figure(figsize=(12, 5))
        plt.plot(subset['date_date'], subset['mean'], marker='o', color='navy',
                 linewidth=1.5, markersize=5, label=f'Posterior Mean {latex_name}')

        plt.fill_between(subset['date_date'], subset['hdi_3%'], subset['hdi_97%'],
                         color='royalblue', alpha=0.15, label='95% HDI')

        plt.ylabel(f'Value of {latex_name}')
        plt.xlabel('Test Month')

        # Handle X-axis tick intervals
        indices = np.arange(0, len(subset), interval)
        plt.xticks(subset['date_date'].iloc[indices],
                   subset['window'].iloc[indices],
                   rotation=45, ha='right')

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"drift_{target_param}.png"), bbox_inches='tight')
        plt.close()

    def draw_ridge_plot(self, df, target_param, save_path, interval=2):
        """
        Generates a ridge plot (Joyplot) to visualize distribution shifts over WFA windows.
        """
        os.makedirs(save_path, exist_ok=True)

        subset = df[df.index == target_param].copy()
        subset['date_date'] = pd.to_datetime(subset['window'])
        subset = subset.sort_values('date_date')

        labels = subset['window'].tolist()
        latex_name = self._get_latex_label(target_param)

        plt.figure(figsize=(12, 6))

        # Determine X-axis range based on HDI
        x_min, x_max = subset['hdi_3%'].min() * 0.9, subset['hdi_97%'].max() * 1.1
        x = np.linspace(x_min, x_max, 500)
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))

        for i, label in enumerate(labels):
            row = subset[subset['window'] == label].iloc[0]
            mu, std = row['mean'], row['sd']

            # Approximate PDF with Normal distribution
            y = norm.pdf(x, mu, std)
            y = y / y.max() * 1.3  # Scaling for ridge overlap effect

            plt.fill_between(x, i, i + y, color=colors[i], alpha=0.7, zorder=len(labels)-i)
            plt.plot(x, i + y, color='black', lw=0.6, zorder=len(labels)-i)

        indices = np.arange(0, len(labels), interval)
        plt.yticks(indices, [labels[idx] for idx in indices])
        plt.ylabel('Test Month')
        plt.xlabel(f'Value of {latex_name}')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.grid(axis='x', linestyle=':', alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"ridge_{target_param}.png"), bbox_inches='tight')
        plt.close()

    def plot_window_diagnostics(self, trace, window_name, save_path):
        """
        Executes a full MCMC diagnostic suite for a specific window.
        Includes Pair Plot, Rank Plot, and Trace Plots.
        """
        os.makedirs(save_path, exist_ok=True)
        print(f"🎨 Generating MCMC diagnostics for window: {window_name}")

        # 1. Pair Plot (Correlation Analysis)
        # Check for trade-offs between parameters (e.g., k vs gamma)
        az.plot_pair(
            trace,
            var_names=self.params,
            kind='kde',
            marginals=True,
            point_estimate='median',
            textsize=14
        )
        plt.gcf().savefig(os.path.join(save_path, f"diag_pair_{window_name}.png"), bbox_inches='tight')
        plt.close()

        # 2. Rank Plot (Chain Mixing Check)
        # Uniform distribution of ranks indicates good convergence
        for param in self.params:
            # Generate rank plot for a single parameter
            az.plot_rank(trace, var_names=[param])

            # Use gcf() to ensure current figure is captured and saved
            plt.gcf().savefig(
                os.path.join(save_path, f"rank_{window_name}_{param}.png"),
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()

        # 3. Trace Plot (Parameter Convergence Check)
        # Visualizing the sampling history for each parameter
        for param in self.params:
            axes = az.plot_trace(trace, var_names=[param])

            # Remove redundant titles for publication quality
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_title("")

            # Use gcf() to ensure current ArviZ figure is captured
            plt.gcf().savefig(
                os.path.join(save_path, f"trace_{window_name}_{param}.png"),
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()

    def plot_backtest_results(self, equity, drawdown, save_path='figures'):
        """
        Plots Equity Curve and Drawdown Profile separately.
        This keeps performance_analyzer focused on pure statistics.
        """
        os.makedirs(save_path, exist_ok=True)

        # 1. Equity Curve
        plt.figure(figsize=(12, 7))
        plt.plot(equity, label='Cumulative Equity', color='blue', linewidth=2)
        plt.title('Strategy Equity Curve')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "equity_curve.png"))
        plt.close()

        # 2. Drawdown Profile
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        plt.title('Drawdown Profile')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "drawdown_profile.png"))
        plt.close()
