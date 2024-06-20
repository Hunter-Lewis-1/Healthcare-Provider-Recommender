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

@pytest.fixture
def sample_ratings_matrix():
    """Create a sample ratings matrix for testing"""
    # 3 users, 4 items
    data = np.array([
        [4.0, 3.0, 0.0, 1.0],
        [0.0, 2.0, 3.0, 4.0],
        [3.0, 0.0, 5.0, 0.0]
    ])
    return pd.DataFrame(data, index=[101, 102, 103], columns=[1, 2, 3, 4])

@pytest.fixture
def sample_providers_df():
    """Create sample provider data for testing"""
    providers = pd.DataFrame({
        'provider_id': [1, 2, 3, 4, 5],
        'quality_score': [80, 90, 70, 85, 95],
        'quality_norm': [0.4, 0.8, 0.0, 0.6, 1.0],
        'cost': [500, 800, 300, 600, 900],
        'cost_norm': [0.33, 0.83, 0.0, 0.5, 1.0],
        'specialty': ['General', 'Cardiology', 'Pediatrics', 'Orthopedics', 'General'],
        'latitude': [40.7, 40.8, 40.6, 40.75, 40.85],
        'longitude': [-74.0, -74.1, -73.9, -74.05, -74.15]
    })
    return providers

def test_collaborative_filter(sample_ratings_matrix):
    """Test SVD-based collaborative filtering"""
    k = 2  # Reduced dimension for testing
    U, sigma, Vt = collaborative_filter(sample_ratings_matrix, k=k)
    
    # Check dimensions
    assert U.shape == (3, k)
    assert sigma.shape == (k,)
    assert Vt.shape == (k, 4)
    
    # Check reconstruction
    reconstruction = U @ np.diag(sigma) @ Vt
    assert reconstruction.shape == (3, 4)

def test_predict_ratings(sample_ratings_matrix):
    """Test rating prediction for a patient"""
    k = 2
    U, sigma, Vt = collaborative_filter(sample_ratings_matrix, k=k)
    
    # Test for existing patient
    predictions = predict_ratings(U, sigma, Vt, 101, sample_ratings_matrix)
    assert len(predictions) == 4
    
    # Test that predictions are reasonable (between 0-5 for ratings)
    assert np.all(predictions >= 0) and np.all(predictions <= 5)

def test_dominates():
    """Test domination function for Pareto optimization"""
    # Create two solutions with provider_id and three objectives
    row1 = pd.Series({
        'provider_id': 1, 
        'quality_norm': 0.9, 
        'cost_norm': -0.2,  # Negative because we minimize cost
        'rating_pred': 4.5
    })
    
    # Dominated solution (worse in all objectives)
    row2 = pd.Series({
        'provider_id': 2,
        'quality_norm': 0.7,
        'cost_norm': -0.4,
        'rating_pred': 4.0
    })
    
    # Non-dominated solution (better in one objective)
    row3 = pd.Series({
        'provider_id': 3,
        'quality_norm': 0.8,
        'cost_norm': -0.1,  # Better cost
        'rating_pred': 4.3
    })
    
    # Test domination
    assert dominates(row1, row2) == True
    assert dominates(row2, row1) == False
    assert dominates(row1, row3) == False
    assert dominates(row3, row1) == False

def test_cluster_providers(sample_providers_df):
    """Test provider clustering"""
    # Test with 2 clusters for simplicity
    labels = cluster_providers(sample_providers_df, n_clusters=2)
    
    assert len(labels) == 5
    assert 'cluster' in sample_providers_df.columns
    assert set(sample_providers_df['cluster'].unique()) == {0, 1}  # Two clusters

def test_pareto_optimize(sample_providers_df):
    """Test Pareto optimization for provider selection"""
    # Create synthetic predicted ratings
    predicted_ratings = np.array([4.5, 3.8, 4.2, 4.0, 3.5])
    
    # Get recommendations
    recommendations = pareto_optimize(sample_providers_df, predicted_ratings, top_n=3)
    
    # Check we get the requested number of recommendations or fewer
    assert len(recommendations) <= 3
    
    # Check that recommendations are in the original dataframe
    assert all(pid in sample_providers_df['provider_id'].values 
               for pid in recommendations['provider_id'].values)
