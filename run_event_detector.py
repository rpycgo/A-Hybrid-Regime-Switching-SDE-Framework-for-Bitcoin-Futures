import argparse
import pandas as pd
from pathlib import Path

from src.utils.configuration_loader import ConfigurationLoader
from src.data.preprocessor import Preprocessor
from src.data.event_detector import EventDetector


def main() -> None:
    """
    Main execution script for event detection.
    Coordinates preprocessing and detection with robust path management.
    """
    # 1. Setup Command Line Argument Parser
    parser = argparse.ArgumentParser(description='Run Microstructure Event Detection Pipeline.')
    parser.add_argument('--input_file', type=str, help='Filename of the raw market data CSV')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output_file', type=str, help='Filename for exported TOML')

    args = parser.parse_args()

    # 2. Initialize Configuration Loader
    config_loader = ConfigurationLoader()
    data_settings = config_loader.get_data_settings()
    event_settings = data_settings.get("event_detection", {})
    collection_settings = data_settings.get("binance_collection", {})

    # 3. Path Resolution using project_root
    # Get only the filename first
    input_filename = (args.input_file or
                      collection_settings.get("output_filename", "btcusdt_future.csv"))

    # Complete absolute path: [Project Root] / data / [Filename]
    input_path = config_loader.project_root / "data" / input_filename

    analysis_start = args.start_date or event_settings.get("analysis_start_date")
    analysis_end = args.end_date or event_settings.get("analysis_end_date")

    if args.output_file:
        event_settings["output_filename"] = args.output_file

    print(f"🚀 Starting Pipeline | Target: {input_path}")
    print(f"📅 Analysis Period: {analysis_start} to {analysis_end}")

    # 4. Load Raw Data
    if not input_path.exists():
        print(f"❌ Error: The file {input_path} was not found.")
        return

    df = pd.read_csv(input_path, parse_dates=True)

    # 5. Step 1: Feature Engineering
    preprocessor = Preprocessor(config_loader)
    print("🔄 Step 1: Generating features and S/R levels...")
    df = preprocessor.calculate_base_features(df)
    df = preprocessor.identify_discovery_sr_levels(df)
    df = preprocessor.calculate_directional_indicators(df)

    # 6. Step 2: Event Detection
    detector = EventDetector(config_loader)
    print("🔍 Step 2: Detecting breakout events...")

    # Filter by date range before detection if needed    
    analysis_df = df.query(f'"{analysis_start}"<= Datetime <= "{analysis_end} 23:59:59"')
    detector.detect_and_save_events(data_frame=analysis_df)


if __name__ == '__main__':
    main()
