import pandas as pd
import numpy as np
from pathlib import Path

from clustering import cluster_providers, get_cluster_stats
from collaborative_filtering import collaborative_filter, predict_ratings
from optimize_pareto import pareto_optimize_fast

class HealthcareRecommender:
    """
    A healthcare provider recommendation system that uses collaborative filtering,
    clustering, and Pareto optimization to generate personalized recommendations.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the recommender system with data from the specified directory.
        
        Args:
            data_dir (str): Directory containing provider and rating data files
        """
        self.data_dir = data_dir
        self.providers_df = None
        self.ratings_df = None
        self.ratings_matrix = None
        self.U = None
        self.sigma = None
        self.Vt = None
        
        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        
        # Apply collaborative filtering
        self._apply_collaborative_filtering()
        
        # Cluster providers
        self._cluster_providers()

    def _load_data(self):
        """Load provider and rating data from CSV files."""
        data_path = Path(self.data_dir)
        
        providers_path = data_path / 'providers_data.csv'
        ratings_path = data_path / 'ratings_data.csv'
        
        if not providers_path.exists() or not ratings_path.exists():
            raise FileNotFoundError(f"Data files not found in {self.data_dir}")
        
        self.providers_df = pd.read_csv(providers_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
    def _preprocess_data(self):
        """Prepare data for recommendation algorithms."""
        # Create user-item matrix for collaborative filtering
        self.ratings_matrix = self.ratings_df.pivot(
            index='patient_id',
            columns='provider_id',
            values='rating'
        ).fillna(0)
        
        # Normalize quality and cost for Pareto optimization
        quality_min = self.providers_df['quality_score'].min()
        quality_max = self.providers_df['quality_score'].max()
        cost_min = self.providers_df['cost'].min()
        cost_max = self.providers_df['cost'].max()
        
        # Add normalized columns (0-1 scale)
        self.providers_df['quality_norm'] = (self.providers_df['quality_score'] - quality_min) / (quality_max - quality_min)
        self.providers_df['cost_norm'] = (self.providers_df['cost'] - cost_min) / (cost_max - cost_min)
    
    def _apply_collaborative_filtering(self, k=50):
        """Apply SVD collaborative filtering to the ratings matrix."""
        # Use SVD implementation from collaborative_filtering module
        self.U, self.sigma, self.Vt = collaborative_filter(self.ratings_matrix, k=k)
    
    def _cluster_providers(self, n_clusters=5):
        """Cluster providers based on their features."""
        # Use clustering implementation from clustering module
        cluster_providers(self.providers_df, n_clusters=n_clusters)
    
    def recommend(self, patient_id, top_n=10):
        """
        Generate recommendations for a specific patient.
        
        Args:
            patient_id (int): ID of the patient to recommend for
            top_n (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Top recommended providers
        """
        # Check if patient exists
        if patient_id not in self.ratings_matrix.index:
            raise ValueError(f"Patient ID {patient_id} not found in ratings data")
            
        # Predict ratings for the patient using collaborative filtering
        predicted_ratings = predict_ratings(
            self.U, 
            self.sigma, 
            self.Vt, 
            patient_id, 
            self.ratings_matrix
        )
        
        # Find providers this patient hasn't rated yet
        rated_providers = self.ratings_df[self.ratings_df['patient_id'] == patient_id]['provider_id'].values
        all_providers = self.providers_df['provider_id'].values
        unrated_mask = ~np.isin(all_providers, rated_providers)
        
        # Get providers the patient hasn't rated
        candidate_providers = self.providers_df[unrated_mask].copy()
        
        # Get the corresponding predicted ratings
        candidate_ratings = predicted_ratings.copy()
        if unrated_mask.any():
            # Filter predicted ratings to only include unrated providers
            provider_indices = np.where(unrated_mask)[0]
            candidate_ratings = predicted_ratings[provider_indices]
        
        # Use Pareto optimization to balance quality, cost, and predicted ratings
        recommendations = pareto_optimize_fast(
            candidate_providers,
            candidate_ratings,
            top_n=top_n
        )
        
        # Add predicted rating to recommendations
        recommendations['predicted_rating'] = candidate_ratings[
            recommendations['provider_id'].apply(
                lambda pid: np.where(candidate_providers['provider_id'] == pid)[0][0]
            )
        ]
        
        # Sort by predicted rating (descending)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        # Return top N recommendations
        return recommendations.head(top_n)
