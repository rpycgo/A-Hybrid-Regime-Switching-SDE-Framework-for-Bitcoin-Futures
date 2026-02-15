import tomllib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.utils.configuration_loader import ConfigurationLoader


class DatasetBuilder:
    """
    Handles mapping of detected events back to the primary DataFrame.
    Prepares 'model_data' segments by tagging them with pre-calculated S/R levels.
    """

    def __init__(self, config_loader: ConfigurationLoader):
        """
        Initializes the builder using the project configuration.
        """
        self.config_loader = config_loader

    def load_events(self, filename: str = "events.toml") -> List[Dict[str, Any]]:
        """
        Parses the detected events from a TOML file and normalizes them.
        Reflects updated schema: start_time, end_time, resistance, support.
        """
        event_path = self.config_loader.project_root / "data" / filename

        if not event_path.exists():
            print(f"⚠️ Warning: Event file not found at {event_path}")
            return []

        with open(event_path, "rb") as f:
            content = tomllib.load(f)

        raw_events = content.get("detected_events", [])
        zones = []

        for index, item in enumerate(raw_events):
            zones.append({
                "id": index + 1,
                "start_time": pd.to_datetime(item["start_time"]),
                "end_time": pd.to_datetime(item["end_time"]),
                "resistance": float(item["resistance"]),
                "support": float(item["support"])
            })

        return zones

    def apply_event_tagging(
        self, df: pd.DataFrame, zones: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Tags the DataFrame with S/R levels and identifies active zones.
        Ensures 'manual_resistance' and 'manual_support' are populated from the event file.
        """
        # Initialize tagging columns
        df["is_in_zone"] = False
        df["zone_id"] = 0
        df["manual_resistance"] = np.nan
        df["manual_support"] = np.nan

        if not pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
            df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Map each event zone onto the primary timeline
        for zone in zones:
            mask = (df["Datetime"] >= zone["start_time"]) & (
                df["Datetime"] <= zone["end_time"]
            )
            df.loc[mask, "is_in_zone"] = True
            df.loc[mask, "zone_id"] = zone["id"]
            df.loc[mask, "manual_resistance"] = zone["resistance"]
            df.loc[mask, "manual_support"] = zone["support"]

        return df

    def slice_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts only the data segments marked for MCMC training.
        """
        if "is_in_zone" not in df.columns:
            raise KeyError("❌ 'is_in_zone' column missing. Run tagging first.")

        # Slice data where is_in_zone is True
        model_data = df[df["is_in_zone"] == True].copy()

        # Final check: Remove rows with essential missing values
        essential_cols = ["log_return", "hybrid_z_score", "manual_resistance"]
        model_data = model_data.dropna(subset=essential_cols)

        print(f"📊 Dataset ready: {len(model_data)} rows prepared for MCMC training.")

        return model_data
