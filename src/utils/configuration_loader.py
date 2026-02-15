import pathlib
import tomllib
from typing import Any, Dict


class ConfigurationLoader:
    """
    A class to load and manage project configurations from TOML files.
    This class centralizes filename management and provides an interface to access 
    data, model, training, and backtest categories.
    """
    def __init__(self, configuration_directory: str = "configs"):
        """
        Initializes the loader with the project root and maps config categories.
        Uses absolute path resolution to ensure reliability in the src/utils/ hierarchy.
        """
        # Resolve project root (src/utils/configuration_loader.py -> project_root)
        self.project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.configuration_directory = self.project_root / configuration_directory

        # Centralized filename management
        self.configuration_filenames = {
            "model": "model_parameters.toml",      # SDE mathematical parameters
            "training": "training_settings.toml",  # MCMC and WFA settings
            "backtest": "backtest_settings.toml",  # Backtest costs and simulation settings
            "data": "data_settings.toml"           # Paths and feature parameters
        }

    def _read_toml_file(self, filename: str) -> Dict[str, Any]:
        """Internal helper to parse TOML into a dictionary."""
        file_path = self.configuration_directory / filename

        if not file_path.exists():
            raise FileNotFoundError(f"❌ Configuration file not found at: {file_path}")

        with open(file_path, "rb") as file:
            return tomllib.load(file)

    def get_model_parameters(self) -> Dict[str, Any]:
        """Retrieve mathematical parameters for the SDE model."""
        return self._read_toml_file(self.configuration_filenames["model"])

    def get_training_settings(self) -> Dict[str, Any]:
        """Retrieve settings for Bayesian inference and WFA schedules."""
        return self._read_toml_file(self.configuration_filenames["training"])

    def get_backtest_settings(self) -> Dict[str, Any]:
        """Retrieve simulation-specific settings for backtesting."""
        return self._read_toml_file(self.configuration_filenames["backtest"])

    def get_data_settings(self) -> Dict[str, Any]:
        """Retrieve configurations for data collection and preprocessing."""
        return self._read_toml_file(self.configuration_filenames["data"])
