"""Implementation of SMOTE for light curve data preprocessing."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def interpolate_vectors(v1, v2, ratio):
    """Creates a new vector by interpolating between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        ratio: Float between 0 and 1, amount of interpolation
        
    Returns:
        Interpolated vector
    """
    return v1 + ratio * (v2 - v1)

def smote_light_curves(light_curves, labels, k_neighbors=5, n_synthetic=None):
    """Applies SMOTE to light curve data.
    
    Args:
        light_curves: Array of shape [n_samples, n_points] containing light curve data
        labels: Binary labels indicating planet (1) vs non-planet (0)
        k_neighbors: Number of nearest neighbors to use
        n_synthetic: Number of synthetic samples to generate. If None, generates
            enough to balance classes.
            
    Returns:
        Tuple of (augmented_light_curves, augmented_labels)
    """
    # Separate minority (planet) and majority (non-planet) classes
    minority_indices = np.where(labels == 1)[0]
    majority_indices = np.where(labels == 0)[0]
    
    if len(minority_indices) == 0:
        return light_curves, labels
    
    # Determine number of synthetic samples needed
    if n_synthetic is None:
        n_synthetic = len(majority_indices) - len(minority_indices)
    
    minority_samples = light_curves[minority_indices]
    
    # Find k nearest neighbors for minority samples
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(minority_samples)
    distances, indices = nn.kneighbors(minority_samples)
    
    synthetic_samples = []
    
    # Generate synthetic samples
    for i in range(len(minority_samples)):
        # Get k nearest neighbors of current sample
        nn_indices = indices[i, 1:]  # Exclude the sample itself
        
        # Generate synthetic samples
        n_needed = min(n_synthetic, k_neighbors - 1)
        for _ in range(n_needed):
            # Randomly select one of the k neighbors
            nn_idx = np.random.choice(nn_indices)
            
            # Generate synthetic sample by interpolation
            ratio = np.random.random()
            synthetic = interpolate_vectors(
                minority_samples[i], 
                minority_samples[nn_idx], 
                ratio
            )
            synthetic_samples.append(synthetic)
            
        n_synthetic -= n_needed
        if n_synthetic <= 0:
            break
    
    if synthetic_samples:
        synthetic_samples = np.stack(synthetic_samples)
        augmented_light_curves = np.vstack([light_curves, synthetic_samples])
        augmented_labels = np.concatenate([
            labels, 
            np.ones(len(synthetic_samples), dtype=labels.dtype)
        ])
        return augmented_light_curves, augmented_labels
    
    return light_curves, labels

def apply_smote_to_tce_data(global_view, local_view, labels, auxiliary_features=None):
    """Applies SMOTE to TCE data including both global and local views.
    
    Args:
        global_view: Array of shape [n_samples, n_global_points]
        local_view: Array of shape [n_samples, n_local_points]
        labels: Binary labels
        auxiliary_features: Optional array of auxiliary features
        
    Returns:
        Tuple of augmented (global_view, local_view, labels, auxiliary_features)
    """
    # Combine global and local views for neighbor calculation
    combined_features = np.hstack([global_view, local_view])
    
    # Apply SMOTE
    augmented_features, augmented_labels = smote_light_curves(
        combined_features, labels)
    
    # Split back into global and local views
    n_global = global_view.shape[1]
    augmented_global = augmented_features[:, :n_global]
    augmented_local = augmented_features[:, n_global:]
    
    if auxiliary_features is not None:
        # For auxiliary features, we'll interpolate them along with the light curves
        minority_indices = np.where(labels == 1)[0]
        n_synthetic = len(augmented_labels) - len(labels)
        
        # Generate synthetic auxiliary features using the same neighbors
        synthetic_aux = []
        for i in range(n_synthetic):
            idx1, idx2 = np.random.choice(minority_indices, 2, replace=False)
            ratio = np.random.random()
            synthetic = interpolate_vectors(
                auxiliary_features[idx1],
                auxiliary_features[idx2],
                ratio
            )
            synthetic_aux.append(synthetic)
            
        if synthetic_aux:
            synthetic_aux = np.stack(synthetic_aux)
            augmented_aux = np.vstack([auxiliary_features, synthetic_aux])
            return augmented_global, augmented_local, augmented_labels, augmented_aux
    
    return augmented_global, augmented_local, augmented_labels, auxiliary_features
