import numpy as np
import pandas as pd

from src.utils.configuration_loader import ConfigurationLoader


class Preprocessor:
    """
    Handles feature engineering for the SDE model and strategy execution.
    The pipeline includes global Z-scores and technical trend indicators.
    """
    def __init__(self, configuration_loader: ConfigurationLoader):
        """
        Initializes the preprocessor with data settings from the loader.
        """
        self.configuration_loader = configuration_loader
        self.settings = self.configuration_loader.get_data_settings().get(
            "event_detection", {}
        )

    def calculate_base_features(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Generates standardized Z-Scores and Hybrid Z-Score.
        Implements denominator defense and noise floor for numerical stability.
        """
        window = self.settings.get("global_window_size", 288)
        noise = self.settings.get("noise_floor", 0.001)
        clipping_limit = self.settings.get("clipping_upper_limit", 6.0)
        minimum_periods = self.settings.get("minimum_periods_standard", 20)
        smoothing_window = self.settings.get("hybrid_smoothing_window", 3)

        # 1. Return Calculations
        data_frame["log_return"] = np.log(
            data_frame["Close"] / data_frame["Close"].shift(1)
        )
        data_frame["absolute_return"] = data_frame["log_return"].abs()

        # 2. Volume Z-Score with Denominator Defense (10% of mean)
        volume_series = data_frame["Volume"]
        rolling_volume_mean = volume_series.rolling(
            window=window, min_periods=minimum_periods
        ).mean()
        rolling_volume_standard_deviation = volume_series.rolling(
            window=window, min_periods=minimum_periods
        ).std()

        minimum_volume_standard_deviation = rolling_volume_mean * 0.1
        rolling_volume_standard_deviation = np.maximum(
            rolling_volume_standard_deviation, minimum_volume_standard_deviation
        )
        data_frame["volume_z_score"] = (
            volume_series - rolling_volume_mean
        ) / rolling_volume_standard_deviation

        # 3. Absolute Return Z-Score with Noise Floor
        return_series = data_frame["absolute_return"]
        rolling_return_mean = return_series.rolling(
            window=window, min_periods=minimum_periods
        ).mean()
        rolling_return_standard_deviation = return_series.rolling(
            window=window, min_periods=minimum_periods
        ).std()

        rolling_return_standard_deviation = np.maximum(
            rolling_return_standard_deviation, noise
        )
        data_frame["absolute_return_z_score"] = (
            (return_series - rolling_return_mean) / rolling_return_standard_deviation
        )

        # 4. Hybrid Z-Score Generation with Clipping
        data_frame["volume_z_score"] = data_frame["volume_z_score"].clip(
            upper=clipping_limit
        )
        data_frame["absolute_return_z_score"] = data_frame[
            "absolute_return_z_score"
        ].clip(upper=clipping_limit)

        raw_hybrid_z = np.maximum(
            data_frame["volume_z_score"], data_frame["absolute_return_z_score"]
        )
        data_frame["hybrid_z_score"] = (
            raw_hybrid_z.rolling(window=smoothing_window, min_periods=1).mean()
        )

        return data_frame

    def identify_discovery_sr_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DISCOVERY: Initial S/R levels used ONLY to find breakout events.
        In the training phase, these are replaced by the 'DatasetBuilder' 
        using values from events.toml.
        """
        threshold = self.settings.get("quiet_regime_threshold", 1.3)
        window = self.settings.get("sr_rolling_window", 288)
        min_periods = self.settings.get("minimum_periods_standard", 20)

        # Identify 'Quiet' state based on Hybrid Z-Score
        df["is_quiet_regime"] = df["hybrid_z_score"] < threshold

        # Record High/Low during Quiet periods
        raw_res = (
            df["High"]
            .where(df["is_quiet_regime"])
            .rolling(window=window, min_periods=min_periods)
            .max()
        )
        raw_sup = (
            df["Low"]
            .where(df["is_quiet_regime"])
            .rolling(window=window, min_periods=min_periods)
            .min()
        )

        # Latching via Forward Fill
        df["manual_resistance"] = raw_res.ffill()
        df["manual_support"] = raw_sup.ffill()

        # Initial Data Defense (Fallback)
        fallback_res = df["High"].rolling(window=window, min_periods=1).max()
        fallback_sup = df["Low"].rolling(window=window, min_periods=1).min()

        df["manual_resistance"] = df["manual_resistance"].fillna(fallback_res)
        df["manual_support"] = df["manual_support"].fillna(fallback_sup)

        return df

    def calculate_strategy_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Adds ADX and Donchian Channels (Dynamic S/R) for strategy entry.
        """
        window = self.settings.get("global_window_size", 288)

        # 1. Dynamic Support and Resistance (Donchian Channel)
        data_frame["dynamic_resistance"] = (
            data_frame["High"].rolling(window=window).max().shift(1)
        )
        data_frame["dynamic_support"] = (
            data_frame["Low"].rolling(window=window).min().shift(1)
        )

        # 2. ADX (Average Directional Index)
        data_frame["adx"] = self._calculate_adx(data_frame)

        return data_frame

    def calculate_directional_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Determines trend direction based on tagged S/R levels.
        Note: Requires manual_resistance/support columns from DatasetBuilder.
        """
        if (
            "manual_resistance" in data_frame.columns
            and "manual_support" in data_frame.columns
        ):
            conditions = [
                (data_frame["Close"] > data_frame["manual_resistance"]),
                (data_frame["Close"] < data_frame["manual_support"]),
            ]
            data_frame["direction_indicator"] = np.select(conditions, [1, -1], default=0)

        return data_frame

    def _calculate_adx(self, data_frame: pd.DataFrame, period: int = 14) -> pd.Series:
        """Internal helper for ADX calculation using Wilder's Smoothing."""
        delta_high = data_frame["High"].diff()
        delta_low = data_frame["Low"].diff()

        plus_directional_movement = np.where(
            (delta_high > delta_low) & (delta_high > 0), delta_high, 0
        )
        minus_directional_movement = np.where(
            (delta_low > delta_high) & (delta_low > 0), delta_low, 0
        )

        true_range = pd.concat([
            data_frame["High"] - data_frame["Low"],
            np.abs(data_frame["High"] - data_frame["Close"].shift(1)),
            np.abs(data_frame["Low"] - data_frame["Close"].shift(1))
        ], axis=1).max(axis=1)

        true_range_sum = true_range.rolling(window=period).sum()
        plus_directional_indicator = 100 * (
            pd.Series(plus_directional_movement).rolling(window=period).sum() 
            / true_range_sum
        )
        minus_directional_indicator = 100 * (
            pd.Series(minus_directional_movement).rolling(window=period).sum() 
            / true_range_sum
        )

        directional_index = 100 * (
            np.abs(plus_directional_indicator - minus_directional_indicator) / 
            (plus_directional_indicator + minus_directional_indicator)
        )

        return directional_index.rolling(window=period).mean()
