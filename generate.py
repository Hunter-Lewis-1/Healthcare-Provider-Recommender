import numpy as np
import pandas as pd
from pathlib import Path

def generate_provider_data(num_providers=10000, num_ratings=100000, seed=42):
    """
    Generate synthetic healthcare provider data and patient ratings.
    
    Args:
        num_providers (int): Number of providers to generate
        num_ratings (int): Number of ratings to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple of (providers_df, ratings_df)
    """
    np.random.seed(seed)
    
    # Generate provider data
    provider_ids = np.arange(1, num_providers + 1)
    quality_scores = np.random.uniform(0, 100, num_providers)
    costs = np.random.lognormal(mean=5, sigma=0.5, size=num_providers)
    specialties = np.random.choice(['General', 'Cardiology', 'Pediatrics', 'Orthopedics'], num_providers)
    latitudes = np.random.uniform(40.5, 40.9, num_providers)  # NYC-ish range
    longitudes = np.random.uniform(-74.25, -73.7, num_providers)
    
    providers_df = pd.DataFrame({
        'provider_id': provider_ids,
        'quality_score': quality_scores,
        'cost': costs,
        'specialty': specialties,
        'latitude': latitudes,
        'longitude': longitudes
    })
    
    # Generate ratings data
    patient_ids = np.random.randint(1, 10001, num_ratings)  # 10k patients
    rated_provider_ids = np.random.choice(provider_ids, num_ratings)
    ratings = np.random.uniform(1, 5, num_ratings)  # 1-5 stars
    
    ratings_df = pd.DataFrame({
        'patient_id': patient_ids,
        'provider_id': rated_provider_ids,
        'rating': ratings
    })
    
    return providers_df, ratings_df

def save_data(providers_df, ratings_df, output_dir='data'):
    """
    Save DataFrames to CSV.
    
    Args:
        providers_df: DataFrame containing provider data
        ratings_df: DataFrame containing ratings data
        output_dir (str): Directory to save the CSV files
    """
    data_path = Path(output_dir)
    data_path.mkdir(exist_ok=True, parents=True)
    
    providers_df.to_csv(data_path / 'providers_data.csv', index=False)
    ratings_df.to_csv(data_path / 'ratings_data.csv', index=False)

if __name__ == '__main__':
    providers, ratings = generate_provider_data()
    save_data(providers, ratings)
    print(f"Generated {len(providers)} providers and {len(ratings)} ratings.")

