import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Model.pipeline import load_data, preprocess_data, get_user_providers

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Sample provider data
    providers = pd.DataFrame({
        'provider_id': [1, 2, 3, 4, 5],
        'quality_score': [80, 90, 70, 85, 95],
        'cost': [500, 800, 300, 600, 900],
        'specialty': ['General', 'Cardiology', 'Pediatrics', 'Orthopedics', 'General'],
        'latitude': [40.7, 40.8, 40.6, 40.75, 40.85],
        'longitude': [-74.0, -74.1, -73.9, -74.05, -74.15]
    })
    
    # Sample ratings data
    ratings = pd.DataFrame({
        'patient_id': [101, 101, 102, 102, 103, 103, 104],
        'provider_id': [1, 2, 1, 3, 2, 4, 5],
        'rating': [4.5, 3.8, 4.2, 4.7, 3.5, 4.9, 4.1]
    })
    
    return providers, ratings

def test_preprocess_data(sample_data):
    """Test preprocessing function"""
    # ...existing code...

def test_get_user_providers(sample_data):
    """Test getting providers rated by a user"""
    # ...existing code...
