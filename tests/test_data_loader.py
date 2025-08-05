import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import load_raw_data, basic_data_info


def test_basic_data_info():
    """Test basic data info function."""
    # Create sample data
    data = pd.DataFrame({
        'A': [1, 2, 3, None],
        'B': ['x', 'y', 'z', 'w'],
        'C': [1.1, 2.2, 3.3, 4.4]
    })
    
    info = basic_data_info(data)
    
    assert info['shape'] == (4, 3)
    assert set(info['columns']) == {'A', 'B', 'C'}
    assert info['missing_values']['A'] == 1
    assert info['missing_values']['B'] == 0
