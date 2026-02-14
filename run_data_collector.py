import argparse

from src.configuration_loader import ConfigurationLoader
from src.data_collector import DataCollector


def run_collection():
    """
    Entry point for the data collection process.
    It merges configurations from a TOML file and CLI arguments.
    """
    # 1. Initialize ConfigurationLoader
    config_loader = ConfigurationLoader()
    toml_settings = config_loader.get_data_settings().get("binance_collection", {})

    # 2. Setup Argparse
    parser = argparse.ArgumentParser(description="Binance Futures Data Collection Script")
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default=toml_settings.get("symbol", "BTCUSDT"),
        help="Trading pair symbol (e.g., ETHUSDT)"
    )
    parser.add_argument(
        "--interval", 
        type=str, 
        default=toml_settings.get("interval", "5m"),
        help="K-line interval (e.g., 1m, 5m, 1h)"
    )
    parser.add_argument(
        "--start", 
        type=str, 
        default=toml_settings.get("start_date", "2024-01-01"),
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=toml_settings.get("output_filename", "market_data.csv"),
        help="Output CSV filename"
    )

    args = parser.parse_args()

    # 3. Execution
    collector = DataCollector(config_loader)
    
    print(f"Starting collection with Symbol: {args.symbol}, Interval: {args.interval}")
    
    market_data = collector.fetch_binance_futures_data(
        symbol=args.symbol,
        interval=args.interval,
        start_str=args.start,
        end_str=None
    )

    collector.save_to_csv(market_data, args.output)
    print(f"Data successfully saved to data/{args.output}")


if __name__ == "__main__":
    run_collection()
