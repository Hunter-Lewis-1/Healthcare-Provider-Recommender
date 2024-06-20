import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_providers(providers_df, n_clusters=5):
    """
    Cluster providers by features.
    
    Args:
        providers_df (pd.DataFrame): Provider data
        n_clusters (int): Number of clusters to create
        
    Returns:
        np.ndarray: Cluster labels for each provider
    """
    # Select features for clustering
    X = providers_df[['quality_norm', 'cost_norm', 'latitude', 'longitude']].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to providers dataframe
    providers_df['cluster'] = labels
    
    return labels

def get_cluster_stats(providers_df):
    """
    Get statistics for each provider cluster.
    
    Args:
        providers_df (pd.DataFrame): Provider data with cluster labels
        
    Returns:
        pd.DataFrame: Statistics for each cluster
    """
    stats = providers_df.groupby('cluster').agg({
        'quality_score': ['mean', 'std'],
        'cost': ['mean', 'std'],
        'provider_id': 'count'
    })
    
    stats.columns = ['quality_mean', 'quality_std', 'cost_mean', 'cost_std', 'count']
    return stats
