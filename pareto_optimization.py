import numpy as np
import pandas as pd

def pareto_optimize(providers_df, predicted_ratings, top_n=10):
    """
    Optimize providers with simplified NSGA-II approach.
    
    Args:
        providers_df (pd.DataFrame): Provider data
        predicted_ratings (np.ndarray): Predicted ratings for providers
        top_n (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: Top recommended providers
    """
    # Create a dataframe with objectives (maximize quality, minimize cost, maximize rating)
    objectives = providers_df[['provider_id', 'quality_norm', 'cost_norm']].copy()
    
    # Add predicted ratings
    objectives['rating_pred'] = predicted_ratings
    
    # Convert cost to negative for maximization (we want to minimize cost)
    objectives['cost_norm'] = -objectives['cost_norm']
    
    # Find non-dominated solutions (Pareto front)
    dominated = np.zeros(len(objectives), dtype=bool)
    
    # O(n²) comparison to find dominated points
    for i in range(len(objectives)):
        if dominated[i]:
            continue
            
        for j in range(len(objectives)):
            if i == j:
                continue
                
            # Check if solution j dominates solution i
            if (dominates(objectives.iloc[j], objectives.iloc[i])):
                dominated[i] = True
                break
    
    # Select non-dominated solutions
    pareto_front = objectives[~dominated]
    
    # Get the top N by predicted rating
    top_indices = pareto_front.nlargest(top_n, 'rating_pred')['provider_id'].values
    
    # Return the provider information for the top recommendations
    recommendations = providers_df[providers_df['provider_id'].isin(top_indices)]
    
    return recommendations

def dominates(row1, row2):
    """
    Check if row1 dominates row2 in a maximization problem.
    A solution dominates another if it's >= in all objectives and > in at least one.
    
    Args:
        row1: Series with objective values
        row2: Series with objective values
        
    Returns:
        bool: True if row1 dominates row2
    """
    # Skip the provider_id column in comparison
    objectives = ['quality_norm', 'cost_norm', 'rating_pred']
    all_greater_equal = all(row1[obj] >= row2[obj] for obj in objectives)
    any_strictly_greater = any(row1[obj] > row2[obj] for obj in objectives)
    
    return all_greater_equal and any_strictly_greater
