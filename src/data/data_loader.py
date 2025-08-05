"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_processed_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save processed data to CSV file.

    Args:
        data (pd.DataFrame): Data to save
        file_path (str): Output file path
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(file_path, index=False)
        logger.info(f"Successfully saved {len(data)} rows to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def basic_data_info(data: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.

    Args:
        data (pd.DataFrame): Input dataframe

    Returns:
        dict: Basic statistics and info
    """
    info = {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "memory_usage": data.memory_usage(deep=True).sum(),
    }
    return info
