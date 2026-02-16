import tomli_w
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional, List, Dict, Any


class EventDetector:
    """Extracts breakout events and handles multi-line logic for clarity."""

    def __init__(self, configuration_loader: Any):
        """
        Initializes the EventDetector with configuration settings.

        Args:
            configuration_loader: Loader instance to fetch data settings.
        """
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
        """
        Detects True/False breakouts and filters them by date range and month boundaries.
        Validates events based on magnitude and persistence before saving.

        Args:
            data_frame: Input DataFrame with DatetimeIndex.
            start_date: Optional start string for analysis.
            end_date: Optional end string for analysis.
        """
        # Fetch settings with defaults
        hz_threshold = self.settings.get("hybrid_z_threshold", 2.0)
        prox_threshold = self.settings.get("sr_proximity_threshold", 0.005)
        mag_threshold = self.settings.get("min_window_magnitude", 0.03)
        persistence_duration = self.settings.get("persistence_duration", 6)
        window_half_hours = self.settings.get("event_window_half_hours", 2)
        output_filename = self.settings.get("output_filename", "events.toml")

        # Determine analysis date range
        analysis_start = start_date or self.settings.get("analysis_start_date")
        analysis_end = end_date or self.settings.get("analysis_end_date")

        # 1. Proximity Calculation
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

        # Date Filtering using Index
        if analysis_start:
            triggers = triggers[triggers.index >= pd.to_datetime(analysis_start)]
        if analysis_end:
            triggers = triggers[triggers.index <= pd.to_datetime(analysis_end)]

        final_events: List[Dict[str, Any]] = []
        last_event_time = None
        interval_seconds = window_half_hours * 2 * 3600

        # 3. Extraction Loop
        for current_time, trigger in triggers.iterrows():
            # Check for overlapping events
            if last_event_time is None or (
                (current_time - last_event_time).total_seconds() > interval_seconds
            ):
                start_window = current_time - timedelta(hours=window_half_hours)
                end_window = current_time + timedelta(hours=window_half_hours)

                # Window data slice
                window_data = data_frame[
                    (data_frame.index >= start_window)
                    & (data_frame.index <= end_window)
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

                # Persistence Validation (Determine if True/False Breakout)
                try:
                    trigger_idx = data_frame.index.get_loc(current_time)
                    future_bars = data_frame.iloc[
                        trigger_idx : trigger_idx + persistence_duration + 1
                    ]
                    
                    resistance = float(trigger["manual_resistance"])
                    support = float(trigger["manual_support"])

                    is_sustained = all(
                        bar["Close"] > resistance or bar["Close"] < support
                        for _, bar in future_bars.iterrows()
                    )

                    # 4. Final Validation: Month Boundary Check (Moved Here)
                    final_end_time = end_window
                    if start_window.month != end_window.month:
                        # Calculate the last valid point of the current month
                        last_day_of_month = (start_window + pd.offsets.MonthEnd(0))
                        adjusted_end = last_day_of_month.replace(
                            hour=23, minute=55, second=0
                        )
                        
                        # Duration check: If less than 2 hours, discard the event
                        duration = adjusted_end - start_window
                        if duration < timedelta(hours=2):
                            print(
                                f"⚠️ Discarded: Valid event candidate at {start_window} "
                                f"crosses month boundary with short duration ({duration})."
                            )
                            # Update last_event_time to prevent immediate re-triggering
                            last_event_time = current_time
                            continue
                        else:
                            print(
                                f"✂️ Truncated: Valid event at {start_window} "
                                f"adjusted to {adjusted_end} due to month boundary."
                            )
                            final_end_time = adjusted_end

                    # 5. Append Validated Event
                    final_events.append(
                        {
                            "start_time": start_window.strftime("%Y-%m-%d %H:%M:%S"),
                            "end_time": final_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "resistance": resistance,
                            "support": support,
                            "event_category": "True Breakout"
                            if is_sustained
                            else "False Breakout",
                            "is_breakout": bool(is_sustained),
                        }
                    )
                    last_event_time = current_time
                    
                except Exception as error:
                    print(f"❌ Error processing event at {current_time}: {error}")
                    continue

        # 6. Export to TOML
        output_path = (
            self.configuration_loader.configuration_directory.parent
            / "data"
            / output_filename
        )
        with open(output_path, "wb") as file:
            tomli_w.dump({"detected_events": final_events}, file)

        print(f"🎯 Analysis Complete: {len(final_events)} events saved.")
