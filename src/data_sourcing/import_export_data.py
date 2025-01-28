from pathlib import Path
import logging
import yaml
import pandas as pd
from typing import Optional, Union, Dict, Any

# Configure logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration manager to handle YAML configuration files.

    Attributes:
        project_root (Path): Root directory of the project
        config_path (Path): Path to the configuration file
        config (Dict): Loaded configuration data
    """

    def __init__(self, env: str = "development") -> None:
        """
        Initialize the configuration manager.

        Args:
            env: Environment name (e.g., "development", "production")

        Raises:
            FileNotFoundError: If config file or project root cannot be found
        """
        self.project_root = self._determine_project_root()
        self.config_path = self._get_config_path(env)
        self.config = self._load_config()

    def _determine_project_root(self) -> Path:
        """
        Determine the project root directory.

        Returns:
            Path: Project root directory
        """
        try:
            return Path(__file__).resolve().parents[2]
        except NameError:
            current_path = Path().resolve()
            while not (current_path / "configs").exists():
                if current_path == current_path.parent:
                    raise FileNotFoundError("Could not find project root with configs directory")
                current_path = current_path.parent
            return current_path

    def _get_config_path(self, env: str) -> Path:
        """
        Get the configuration file path.

        Args:
            env: Environment name

        Returns:
            Path: Configuration file path
        """
        config_path = self.project_root / "configs" / f"{env}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return config_path

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dict: Configuration data
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the configuration using dot notation.

        Args:
            key: Configuration key using dot notation (e.g., "paths.raw_data")
            default: Default value if key not found

        Returns:
            Configuration value or default if not found
        """
        value = self.config
        for k in key.split('.'):
            value = value.get(k, default)
            if value is None:
                break
        return value

def get_data(
    filename: str,
    data_type: str = "raw_data",
    env: str = "development"
) -> pd.DataFrame:
    """
    Load a data file dynamically based on the configuration.

    Args:
        filename: Name of the data file
        data_type: Type of data (raw_data, processed_data, etc.)
        env: Environment (development, production, etc.)

    Returns:
        pd.DataFrame: Loaded data

    Raises:
        FileNotFoundError: If the data file cannot be found
    """
    config_manager = ConfigManager(env)
    base_path = config_manager.project_root / config_manager.get(f"paths.{data_type}")
    file_path = base_path / filename

    logger.debug(f"Loading data from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)

def save_data(
    dataframe: pd.DataFrame,
    filename: str,
    data_type: str = "processed_data",
    env: str = "development"
) -> None:
    """
    Save a DataFrame to a specified location based on the configuration.

    Args:
        dataframe: The DataFrame to save
        filename: Name of the output file
        data_type: Type of data directory
        env: Environment name

    Raises:
        ValueError: If input validation fails
        KeyError: If data_type is not found in config
    """
    # Input validation
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")

    config_manager = ConfigManager(env)
    base_path = config_manager.project_root / config_manager.get(f"paths.{data_type}")

    if base_path is None:
        raise KeyError(f"Missing configuration for paths.{data_type}")

    # Ensure directory exists and save file
    base_path.mkdir(parents=True, exist_ok=True)
    file_path = base_path / filename
    dataframe.to_csv(file_path, index=False)

    logger.info(
        f"Data saved successfully!\n"
        f"Environment: {env}\n"
        f"Data Type: {data_type}\n"
        f"File Path: {file_path}\n"
        f"Rows Saved: {len(dataframe)}"
    )