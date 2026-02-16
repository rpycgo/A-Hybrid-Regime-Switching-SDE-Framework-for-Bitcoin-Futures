import tomli_w
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional

from src.utils.configuration_loader import ConfigurationLoader


class EventDetector:
    """Extracts breakout events and handles multi-line logic for clarity."""
    def __init__(self, configuration_loader: ConfigurationLoader):
        self.configuration_loader = configuration_loader
        self.settings = self.configuration_loader.get_data_settings().get(
            "event_detection", {}
        )

    def detect_and_save_events(
        self,
        data_frame: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """Detects True/False breakouts and filters them by date range."""
        if not pd.api.types.is_datetime64_any_dtype(data_frame["Datetime"]):
            data_frame["Datetime"] = pd.to_datetime(data_frame["Datetime"], utc=True).dt.tz_localize(None)

        # Fetch settings with defaults
        hz_threshold = self.settings.get("hybrid_z_threshold", 2.0)
        prox_threshold = self.settings.get("sr_proximity_threshold", 0.005)
        mag_threshold = self.settings.get("min_window_magnitude", 0.03)
        p_duration = self.settings.get("persistence_duration", 6)
        win_half_hours = self.settings.get("event_window_half_hours", 2)
        output_name = self.settings.get("output_filename", "events.toml")

        # Determine analysis date range
        analysis_start = start_date or self.settings.get("analysis_start_date")
        analysis_end = end_date or self.settings.get("analysis_end_date")

        # 1. Proximity Calculation (Multi-line)
        data_frame["prox_res"] = (
            np.abs(data_frame["Close"] - data_frame["manual_resistance"])
            / data_frame["manual_resistance"]
        )
        data_frame["prox_sup"] = (
            np.abs(data_frame["Close"] - data_frame["manual_support"])
            / data_frame["manual_support"]
        )

        # 2. Trigger Identification
        condition = (data_frame["hybrid_z_score"] > hz_threshold) & (
            (data_frame["prox_res"] < prox_threshold)
            | (data_frame["prox_sup"] < prox_threshold)
        )
        triggers = data_frame[condition].copy()

        # Date Filtering
        if analysis_start:
            triggers = triggers[triggers["Datetime"] >= pd.to_datetime(analysis_start)]
        if analysis_end:
            triggers = triggers[triggers["Datetime"] <= pd.to_datetime(analysis_end)]

        final_events = []
        last_event_time = None
        interval_seconds = win_half_hours * 2 * 3600

        # 3. Extraction Loop
        for _, trigger in triggers.iterrows():
            current_time = trigger["Datetime"]

            if last_event_time is None or (
                (current_time - last_event_time).total_seconds() > interval_seconds
            ):
                start_window = current_time - timedelta(hours=win_half_hours)
                end_window = current_time + timedelta(hours=win_half_hours)

                window_data = data_frame[
                    (data_frame["Datetime"] >= start_window)
                    & (data_frame["Datetime"] <= end_window)
                ]
                if window_data.empty:
                    continue

                # Magnitude Validation
                opening_price = window_data.iloc[0]["Open"]
                highest, lowest = window_data["High"].max(), window_data["Low"].min()
                max_move = max(
                    (highest - opening_price) / opening_price,
                    (opening_price - lowest) / opening_price,
                )
                if max_move < mag_threshold:
                    continue

                # Persistence Validation
                try:
                    trigger_idx = data_frame.index[
                        data_frame["Datetime"] == current_time
                    ].tolist()[0]
                    future_bars = data_frame.iloc[
                        trigger_idx : trigger_idx + p_duration + 1
                    ]
                    res, sup = float(trigger["manual_resistance"]), float(
                        trigger["manual_support"]
                    )

                    is_sustained = all(
                        bar["Close"] > res or bar["Close"] < sup
                        for _, bar in future_bars.iterrows()
                    )

                    final_events.append(
                        {
                            "start_time": start_window.strftime("%Y-%m-%d %H:%M:%S"),
                            "end_time": end_window.strftime("%Y-%m-%d %H:%M:%S"),
                            "resistance": res,
                            "support": sup,
                            "event_category": "True Breakout"
                            if is_sustained
                            else "False Breakout",
                            "is_breakout": bool(is_sustained),
                        }
                    )
                    last_event_time = current_time
                except Exception:
                    continue

        # 4. Export
        output_path = (
            self.configuration_loader.configuration_directory.parent
            / "data"
            / output_name
        )
        with open(output_path, "wb") as file:
            tomli_w.dump({"detected_events": final_events}, file)

        print(f"🎯 Analysis Complete: {len(final_events)} events saved.")
