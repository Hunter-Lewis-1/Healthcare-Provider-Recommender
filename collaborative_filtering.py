import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def collaborative_filter(ratings_matrix, k=50):
    """
    Apply SVD for collaborative filtering.
    
    Args:
        ratings_matrix (pd.DataFrame): User-item matrix of ratings
        k (int): Number of latent factors
        
    Returns:
        tuple (U, sigma, Vt): Factors from SVD
    """
    # Convert to sparse matrix for efficiency
    R = csr_matrix(ratings_matrix.values)
    
    # Apply SVD - efficient O(nk) implementation
    U, sigma, Vt = svds(R, k=k)
    
    # Sort by singular values in descending order
    idx = np.argsort(-sigma)
    U = U[:, idx]
    sigma = sigma[idx]
    Vt = Vt[idx, :]
    
    return U, sigma, Vt

def predict_ratings(U, sigma, Vt, patient_id, ratings_matrix):
    """
    Predict ratings for a patient.
    
    Args:
        U: User latent factors
        sigma: Singular values
        Vt: Item latent factors transposed
        patient_id (int): ID of the patient
        ratings_matrix (pd.DataFrame): Original ratings matrix
        
    Returns:
        np.ndarray: Predicted ratings for all providers
    """
    try:
        # Get the index of the patient in the ratings matrix
        idx = ratings_matrix.index.get_loc(patient_id)
        
        # Calculate predicted ratings using matrix multiplication
        predicted = U[idx, :] @ np.diag(sigma) @ Vt
        
        return predicted
    except KeyError:
        # Handle case where patient_id is not in the ratings matrix
        print(f"Patient ID {patient_id} not found in ratings data")
        return np.zeros(Vt.shape[1])
