import yaml
from pathlib import Path

def get_data(filename, data_type="raw_data"):
    # Load the YAML configuration
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Construct the path to the data file dynamically using the variable
    file_path = Path(f"../{config['paths'][data_type]}/{filename}")

    return file_path
