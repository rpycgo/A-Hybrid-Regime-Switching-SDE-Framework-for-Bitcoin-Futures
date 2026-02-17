from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from src.utils.configuration_loader import ConfigurationLoader


class SdeModeler:
    """
    Implements a Hybrid Stochastic Differential Equation (SDE) model using Bayesian inference.
    
    Mathematical Structure (Text Format):
    1. Regime Weight: w_t = 1 / (1 + exp(-k * (Z_t - gamma)))
    2. Combined Drift: mu_t = (1 - w_t) * (-kappa * r_t) + w_t * (alpha * d_t)
    3. Combined Diffusion: sigma_t = (1 - w_t) * sigma_0 + w_t * sigma_1
    4. Discrete Transition (Euler-Maruyama): 
       r_t ~ Normal(mu_t * dt, sigma_t * sqrt(dt))
    """
    def __init__(self, config_loader: ConfigurationLoader):
        """
        Initializes the modeler and loads settings from external TOML files.
        """
        self.config_loader = config_loader

        # Load priors from model_parameters.toml
        self.priors = self.config_loader.get_model_parameters().get("sde_priors", {})

        # Load sampling and initial settings from training_settings.toml
        training_data = self.config_loader.get_training_settings()
        self.sampling_settings = training_data.get("mcmc_settings", {})
        self.initial_values = training_data.get("initial_values", {})

        # Load time-step related configurations
        self.data_settings = self.config_loader.get_data_settings().get("event_detection", {})

    def estimate_parameters(
        self,
        z_values: np.ndarray,
        returns_scaled: np.ndarray,
        direction: np.ndarray
    ) -> Tuple[Optional[az.InferenceData], Optional[pd.DataFrame], Optional[Dict[str, float]]]:
        """
        Estimates SDE parameters using the NUTS sampler with proper time scaling (dt).
        
        Args:
            z_values: Input Z-Score vector (Z_t).
            returns_scaled: Scaled log-returns (r_t * 100).
            direction: Trend direction indicator (d_t).
            
        Returns:
            A tuple containing MCMC trace, summary statistics, and parameter estimates.
        """
        # Fetch MCMC execution settings
        draws = self.sampling_settings.get("draws", 2000)
        tune = self.sampling_settings.get("tune", 1000)
        chains = self.sampling_settings.get("chains", 4)
        target_accept = self.sampling_settings.get("target_accept", 0.99)
        sampler_engine = self.sampling_settings.get("nuts_sampler", "numpyro")
        random_seed = self.sampling_settings.get("random_seed")
        show_progress = self.sampling_settings.get("show_progress_bar", True)

        # Time-step calculation: dt = 1 / global_window_size
        global_window = self.data_settings.get("global_window_size", 288)
        time_step = 1.0 / global_window

        # --- FIX: Data Slicing to eliminate Look-ahead Bias ---
        # We use information at time 't' to predict the outcome at time 't+1'
        z_values = z_values[:-1]            # Z_t
        return_lag = returns_scaled[:-1]    # r_t
        direction_lag = direction[:-1]      # d_t
        return_target = returns_scaled[1:]  # Target: r_{t+1}

        # --- UPDATE: Directional Components for Asymmetric Drift ---
        # Splitting trend direction to allow independent learning of long and short dynamics
        d_long = np.where(direction_lag > 0, 1.0, 0.0)
        d_short = np.where(direction_lag < 0, -1.0, 0.0)

        with pm.Model() as hybrid_sde_model:
            # --- 1. Drift Prior Definitions ---
            kappa = pm.HalfNormal("kappa", sigma=self.priors.get("kappa_sigma", 1.0))
            alpha_long = pm.Normal(
                "alpha_long", 
                mu=self.priors.get("alpha_mu", 50.0), 
                sigma=self.priors.get("alpha_sigma", 10.0)
            )
            alpha_short = pm.Normal(
                "alpha_short", 
                mu=self.priors.get("alpha_mu", 50.0), 
                sigma=self.priors.get("alpha_sigma", 10.0)
            )

            # --- 2. Diffusion Prior Definitions (sigma_0: Quiet, sigma_1: Breakout) ---
            sigma_0 = pm.HalfNormal(
                "sigma_0", 
                sigma=self.priors.get("sigma_0_sigma", 1.0)
            )
            sigma_1 = pm.TruncatedNormal(
                "sigma_1", 
                mu=self.priors.get("sigma_1_mu", 10.0),
                sigma=self.priors.get("sigma_1_sigma", 2.0),
                lower=self.priors.get("sigma_1_lower", 2.0),
            )

            # --- 3. Regime Switching Parameters (gamma: Threshold, k: Speed) ---
            gamma = pm.TruncatedNormal(
                "gamma", 
                mu=self.priors.get("gamma_mu", 2.0),
                sigma=self.priors.get("gamma_sigma", 0.5),
                lower=self.priors.get("gamma_lower", 1.0),
            )
            k = pm.HalfNormal("k", sigma=self.priors.get("k_sigma", 2.0))

            # --- 4. Transition and Likelihood Construction ---
            # Compute regime transition weight (sigmoid)
            w_t = pm.Deterministic(
                "regime_weight", pm.math.sigmoid(k * (z_values - gamma))
            )

            # Combined drift and diffusion
            drift_breakout = alpha_long*d_long + alpha_short*d_short
            mu = (1 - w_t)*(-kappa * return_lag) + w_t*drift_breakout
            sigma = (1 - w_t)*sigma_0 + w_t*sigma_1

            # Likelihood based on Euler-Maruyama discretization
            # returns ~ Normal(mu * dt, sigma * sqrt(dt))
            pm.Normal(
                "likelihood",
                mu=mu * time_step,
                sigma=(sigma + 1e-6) * np.sqrt(time_step),
                observed=return_target,
            )

            # --- 5. MCMC Sampling Execution ---
            try:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    nuts_sampler=sampler_engine,
                    initvals=self.initial_values,
                    random_seed=random_seed,
                    progressbar=show_progress
                )

                param_names = [
                    "alpha_long", "alpha_short", "kappa",
                    "gamma", "k", "sigma_0", "sigma_1",
                ]
                summary_df = az.summary(trace, var_names=param_names)

                return trace, summary_df, summary_df["mean"].to_dict()

            except Exception as sampling_error:
                print(f"⚠️ MCMC Sampling Failed: {sampling_error}")
                return None, None, None
