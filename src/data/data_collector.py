import time
import pathlib
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List, Any

from src.utils.configuration_loader import ConfigurationLoader


class DataCollector:
    """
    A class to collect historical and real-time market data from Binance Futures.
    Supports paginated fetching to retrieve large datasets efficiently.
    """
    def __init__(self, configuration_loader: ConfigurationLoader):
        """
        Initialize the DataCollector with configuration settings.
        """
        self.configuration_loader = configuration_loader
        self.data_settings = self.configuration_loader.get_data_settings()

        # Binance Futures API Endpoint
        self.base_url = "https://fapi.binance.com/fapi/v1/klines"

    def fetch_binance_futures_data(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance Futures with pagination.
        Defaults are pulled from the configuration file if not provided.
        """
        # 1. Fallback to configuration settings if arguments are missing
        symbol = symbol or self.data_settings.get("binance_collection", {}).get("symbol", "BTCUSDT")
        interval = interval or self.data_settings.get("binance_collection", {}).get("interval", "5m")
        start_str = start_date or self.data_settings.get("binance_collection", {}).get("start_date", "2024-01-01")

        # 2. Time Range Setup
        start_datetime = datetime.strptime(start_str, "%Y-%m-%d")
        current_timestamp = int(start_datetime.timestamp() * 1000)

        if end_date:
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            end_timestamp = int(end_datetime.timestamp() * 1000)
            target_display_message = f"{end_date} 00:00"
        else:
            end_timestamp = int(time.time() * 1000)
            target_display_message = "Now"

        data_list: List[List[Any]] = []
        print(f"🚀 [{symbol}] Starting collection for interval: {interval}")
        print(f"📅 Period: {start_str} ~ {target_display_message}")

        # 3. Paginated Data Retrieval
        while current_timestamp < end_timestamp:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_timestamp,
                "limit": 1500
            }

            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    print("⚠️ No more data available (Empty response).")
                    break

                filtered_data = [candle for candle in data if candle[0] < end_timestamp]
                if not filtered_data:
                    break

                data_list.extend(filtered_data)

                # Update timestamp for next iteration
                last_candle_close_time = data[-1][6]
                current_timestamp = last_candle_close_time + 1

                # Logging Progress
                last_date_progress = datetime.fromtimestamp(
                    data[-1][0] / 1000
                ).strftime('%Y-%m-%d %H:%M')
                print(f"   ... Collected up to {last_date_progress} ({len(data_list)} candles)")

                if len(data) != len(filtered_data):
                    break

                time.sleep(0.1)  # Rate limit protection

            except Exception as error:
                print(f"❌ Error occurred: {error}")
                time.sleep(1)
                continue

        # 4. Data Frame Processing
        ohlcv_columns = [
            "Open_Time", "Open", "High", "Low", "Close", "Volume",
            "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
            "Taker_Buy_Base_Asset_Volume", "Taker_Buy_Quote_Asset_Volume", "Ignore"
        ]
        data_frame = pd.DataFrame(data_list, columns=ohlcv_columns)

        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        data_frame[numeric_columns] = data_frame[numeric_columns].apply(
            pd.to_numeric, errors='coerce'
        )

        data_frame['Datetime'] = pd.to_datetime(data_frame['Open_Time'], unit='ms')
        data_frame.set_index('Datetime', inplace=True)

        return data_frame[numeric_columns]

    def save_to_csv(self, data_frame: pd.DataFrame, filename: Optional[str] = None) -> None:
        """
        Save the processed dataframe to the project's data directory.
        """
        # Resolve path using the new project_root attribute from Loader
        filename = filename or self.data_settings.get("binance_collection", {}).get("output_filename", "data.csv")
        save_path = self.configuration_loader.project_root / "data" / filename

        # Ensure target directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data_frame.to_csv(save_path)
        print(f"💾 File saved successfully at: {save_path}")
