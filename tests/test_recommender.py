import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Model.collaborative_filtering import collaborative_filter, predict_ratings
from Model.clustering import cluster_providers
from Model.pareto_optimization import pareto_optimize, dominates
from Model.recommender import HealthcareRecommender

@pytest.fixture
def sample_ratings_matrix():
    """Create a sample ratings matrix for testing"""
    # ...existing code...

@pytest.fixture
def sample_data_files(tmp_path):
    """Create temporary data files for testing"""
    # Create a temporary directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create provider data
    providers = pd.DataFrame({
        'provider_id': [1, 2, 3, 4, 5],
        'quality_score': [80, 90, 70, 85, 95],
        'cost': [500, 800, 300, 600, 900],
        'specialty': ['General', 'Cardiology', 'Pediatrics', 'Orthopedics', 'General'],
        'latitude': [40.7, 40.8, 40.6, 40.75, 40.85],
        'longitude': [-74.0, -74.1, -73.9, -74.05, -74.15]
    })
    
    # Create ratings data
    ratings = pd.DataFrame({
        'patient_id': [101, 101, 102, 102, 103, 103, 104],
        'provider_id': [1, 2, 1, 3, 2, 4, 5],
        'rating': [4.5, 3.8, 4.2, 4.7, 3.5, 4.9, 4.1]
    })
    
    # Save to CSV files
    providers.to_csv(data_dir / "providers_data.csv", index=False)
    ratings.to_csv(data_dir / "ratings_data.csv", index=False)
    
    return tmp_path

def test_collaborative_filter(sample_ratings_matrix):
    """Test SVD-based collaborative filtering"""
    # ...existing code...

def test_dominates():
    """Test domination function for Pareto optimization"""
    # ...existing code...

def test_cluster_providers():
    """Test provider clustering"""
    # ...existing code...

def test_recommender_initialization(sample_data_files):
    """Test initializing the recommender"""
    # Initialize recommender with sample data
    recommender = HealthcareRecommender(data_dir=sample_data_files / "data", k=2, n_clusters=2)
    
    # Check that data was loaded correctly
    assert recommender.providers_df is not None
    assert recommender.ratings_df is not None
    assert recommender.ratings_matrix is not None
    assert recommender.U is not None
    assert recommender.sigma is not None
    assert recommender.Vt is not None
    
    # Check that providers were clustered
    assert 'cluster' in recommender.providers_df.columns

def test_recommender_recommend(sample_data_files):
    """Test recommendation generation"""
    # Initialize recommender with sample data
    recommender = HealthcareRecommender(data_dir=sample_data_files / "data", k=2, n_clusters=2)
    
    # Get recommendations for a patient
    recommendations = recommender.recommend(101, top_n=3)
    
    # Check that we got recommendations
    assert len(recommendations) <= 3
    assert all(col in recommendations.columns for col in ['provider_id', 'quality_score', 'cost', 'specialty', 'predicted_rating'])
    
    # Check that predicted_rating is present and reasonable
    assert all(0 <= rating <= 5 for rating in recommendations['predicted_rating'])

def test_recommender_invalid_patient(sample_data_files):
    """Test handling invalid patient IDs"""
    # Initialize recommender with sample data
    recommender = HealthcareRecommender(data_dir=sample_data_files / "data", k=2, n_clusters=2)
    
    # Try to get recommendations for a non-existent patient
    with pytest.raises(Exception):
        # This should raise an exception because patient 999 doesn't exist
        recommender.recommend(999, top_n=3)
