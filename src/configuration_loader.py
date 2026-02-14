import pathlib
import tomllib
from typing import Any, Dict


class ConfigurationLoader:
    """
    A class to load and manage project configurations from TOML files.
    This class centralizes filename management and provides an interface to access 
    different configuration categories.
    """

    def __init__(self, configuration_directory: str = "configs"):
        """
        Initialize the loader with the directory path and map configuration types to filenames.

        Args:
            configuration_directory (str): The relative or absolute path to the configurations folder.
        """
        self.configuration_directory = pathlib.Path(configuration_directory)
        
        # Centralized filename management to avoid hardcoding in multiple methods
        self.configuration_filenames = {
            "model": "model_parameters.toml",
            "trading": "trading_settings.toml",
            "data": "data_settings.toml"
        }

    def _read_toml_file(self, filename: str) -> Dict[str, Any]:
        """
        Internal helper method to read a TOML file and convert it into a Python dictionary.

        Args:
            filename (str): The name of the file to be read.

        Returns:
            Dict[str, Any]: The parsed configuration data.

        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
        """
        file_path = self.configuration_directory / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "rb") as file:
            return tomllib.load(file)

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Retrieve parameters related to the SDE model and MCMC sampling.

        Returns:
            Dict[str, Any]: Model-specific configuration values.
        """
        return self._read_toml_file(self.configuration_filenames["model"])

    def get_trading_settings(self) -> Dict[str, Any]:
        """
        Retrieve settings for the trading strategy, including risk limits and costs.

        Returns:
            Dict[str, Any]: Trading strategy configuration values.
        """
        return self._read_toml_file(self.configuration_filenames["trading"])

    def get_data_settings(self) -> Dict[str, Any]:
        """
        Retrieve settings for data paths and feature engineering parameters.

        Returns:
            Dict[str, Any]: Data and feature processing configuration values.
        """
        return self._read_toml_file(self.configuration_filenames["data"])
