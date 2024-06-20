import numpy as np
import pandas as pd
from pathlib import Path

def load_data(data_dir='data'):
    """
    Load providers and ratings CSVs.
    
    Args:
        data_dir (str): Directory containing the CSV files
    
    Returns:
        tuple (providers_df, ratings_df)
    """
    data_path = Path(data_dir)
    providers_df = pd.read_csv(data_path / 'providers_data.csv')
    ratings_df = pd.read_csv(data_path / 'ratings_data.csv')
    
    return providers_df, ratings_df

def preprocess_data(providers_df, ratings_df):
    """
    Preprocess data for modeling.
    
    Args:
        providers_df: DataFrame containing provider data
        ratings_df: DataFrame containing ratings data
    
    Returns:
        tuple (processed_providers, ratings_matrix)
    """
    # Clean data
    providers_df.dropna(inplace=True)
    ratings_df.dropna(inplace=True)
    
    # Normalize provider features
    providers_df['quality_norm'] = (providers_df['quality_score'] - providers_df['quality_score'].min()) / \
                                  (providers_df['quality_score'].max() - providers_df['quality_score'].min())
    
    providers_df['cost_norm'] = (providers_df['cost'] - providers_df['cost'].min()) / \
                               (providers_df['cost'].max() - providers_df['cost'].min())
    
    # Create user-item matrix for collaborative filtering
    ratings_matrix = ratings_df.pivot(index='patient_id', columns='provider_id', values='rating').fillna(0)
    
    return providers_df, ratings_matrix

def get_user_providers(ratings_df, patient_id):
    """
    Get providers rated by a specific patient.
    
    Args:
        ratings_df: DataFrame containing ratings data
        patient_id (int): ID of the patient
        
    Returns:
        List of provider IDs rated by the patient
    """
    user_ratings = ratings_df[ratings_df['patient_id'] == patient_id]
    return user_ratings['provider_id'].tolist()
