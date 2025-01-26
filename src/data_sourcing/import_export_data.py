import yaml
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data(filename, data_type="raw_data"):
    # Load the YAML configuration
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Construct the path to the data file dynamically
    file_path = Path(f"../{config['paths'][data_type]}/{filename}")

    # Verify the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logging.info(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    # Read the file into a Pandas DataFrame
    return df

def save_data(dataframe, filename, data_type="processed_data"):
    # Load the YAML configuration
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Construct the path to the save location dynamically
    file_path = Path(f"../{config['paths'][data_type]}/{filename}")

    # Ensure the target directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to the specified location
    dataframe.to_csv(file_path, index=False)
    logging.info(f"Data saved to: {file_path}")

