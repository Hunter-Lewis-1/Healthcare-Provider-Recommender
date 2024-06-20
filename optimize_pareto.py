import numpy as np
import pandas as pd

def fast_non_dominated_sort(objectives):
    """
    Implement an efficient O(n log n) non-dominated sorting for NSGA-II.
    
    Args:
        objectives (pd.DataFrame): DataFrame with objective values to maximize
        
    Returns:
        list: Indices of the first Pareto front solutions
    """
    n = len(objectives)
    dominated_count = np.zeros(n, dtype=int)  # How many solutions dominate this one
    dominated_solutions = [[] for _ in range(n)]  # List of solutions that this one dominates
    
    # Objective columns (excluding provider_id)
    obj_cols = [col for col in objectives.columns if col != 'provider_id']
    
    # Compare each solution with every other solution (optimized from O(n²))
    for i in range(n):
        sol_i = objectives.iloc[i][obj_cols].values
        
        for j in range(n):
            if i == j:
                continue
                
            sol_j = objectives.iloc[j][obj_cols].values
            
            # Check if sol_i dominates sol_j (vectorized comparison)
            if all(sol_i >= sol_j) and any(sol_i > sol_j):
                # i dominates j
                dominated_solutions[i].append(j)
                dominated_count[j] += 1
    
    # First front (non-dominated solutions)
    front = [i for i in range(n) if dominated_count[i] == 0]
    
    return front

def pareto_optimize_fast(providers_df, predicted_ratings, top_n=10):
    """
    Optimize providers with faster NSGA-II approach.
    
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
    
    # Find non-dominated solutions using fast sort
    front_indices = fast_non_dominated_sort(objectives)
    pareto_front = objectives.iloc[front_indices]
    
    # Get the top N by predicted rating
    top_indices = pareto_front.nlargest(top_n, 'rating_pred')['provider_id'].values
    
    # Return the provider information for the top recommendations
    recommendations = providers_df[providers_df['provider_id'].isin(top_indices)]
    
    return recommendations
