import pandas as pd
import requests
import pathlib
import time
from datetime import datetime
from typing import Optional, List, Any

from src.configuration_loader import ConfigurationLoader


class DataCollector:
    """
    A class to collect historical and real-time market data from Binance Futures.
    It supports paginated fetching to retrieve large datasets over a specified time range.
    """

    def __init__(self, configuration_loader: ConfigurationLoader):
        """
        Initialize the DataCollector with configuration settings.
        """
        self.configuration_loader = configuration_loader
        self.data_settings = self.configuration_loader.get_data_settings()
        self.base_url = "https://fapi.binance.com/fapi/v1/klines"

    def fetch_binance_futures_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "5m",
        start_str: str = "2024-01-01",
        end_str: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance Futures with pagination.

        Returns:
            pd.DataFrame: A dataframe containing Open, High, Low, Close, Volume.
        """
        # 1. Convert start time to milliseconds timestamp
        start_datetime = datetime.strptime(start_str, "%Y-%m-%d")
        current_timestamp = int(start_datetime.timestamp() * 1000)

        # 2. Set end time logic
        if end_str:
            end_datetime = datetime.strptime(end_str, "%Y-%m-%d")
            end_timestamp = int(end_datetime.timestamp() * 1000)
            target_message = f"{end_str} 00:00"
        else:
            end_timestamp = int(time.time() * 1000)
            target_message = "Now"

        data_list: List[List[Any]] = []

        print(f"🚀 [{symbol}] Starting collection for interval: {interval}")
        print(f"📅 Period: {start_str} ~ {target_message}")

        while current_timestamp < end_timestamp:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_timestamp,
                "limit": 1500
            }

            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data:
                    print("⚠️ No more data available (Empty response).")
                    break

                # Filter data that exceeds end_timestamp
                filtered_data = [
                    candle for candle in data if candle[0] < end_timestamp
                ]

                if not filtered_data:
                    print("🏁 Reached the specified end time.")
                    break

                data_list.extend(filtered_data)

                # Update current_timestamp to the next candle's start time
                last_close_time = data[-1][6]
                current_timestamp = last_close_time + 1

                # Print progress
                last_date_str = datetime.fromtimestamp(data[-1][0] / 1000).strftime('%Y-%m-%d %H:%M')
                print(f"   ... Collected up to {last_date_str} (Total: {len(data_list)} candles)")

                if len(data) != len(filtered_data):
                    print("🏁 Reached the exact end date. Stopping.")
                    break

                time.sleep(0.1)  # Rate limit protection

            except Exception as error:
                print(f"❌ Error occurred: {error}")
                time.sleep(1)
                continue

        # 3. Final data processing
        columns = [
            "Open_Time", "Open", "High", "Low", "Close", "Volume",
            "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
            "Taker_Buy_Base_Asset_Volume", "Taker_Buy_Quote_Asset_Volume", "Ignore"
        ]

        df = pd.DataFrame(data_list, columns=columns)

        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        df['Datetime'] = pd.to_datetime(df['Open_Time'], unit='ms')
        df.set_index('Datetime', inplace=True)

        final_df = df[["Open", "High", "Low", "Close", "Volume"]]

        print(f"✅ Collection finished! Total candles: {len(final_df)}")
        return final_df

    def save_to_csv(self, data_frame: pd.DataFrame, filename: str) -> None:
        """
        Save the processed dataframe to the data directory.
        """
        # Ensure the path is handled correctly using pathlib
        project_root = self.configuration_loader.configuration_directory.parent
        save_path = project_root / "data" / filename
        
        # Create directory if it doesn't exist (safety measure)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_frame.to_csv(save_path)
