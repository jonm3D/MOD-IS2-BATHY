def custom_kde(data, method='silverman', grid_points=1000, grid_min=None, grid_max=None, mirror=(None, None)):
    """
    A function for kernel density estimation with various bandwidth selection methods
    and optional data mirroring to handle edge effects.
    
    Parameters
    ----------
    data : array-like
        1D array of data points.
    method : str or float, default='silverman'
        Method for bandwidth selection or a direct bandwidth value.
        String options are:
        - 'scott': Scott's rule (suitable for nearly Gaussian distributions)
        - 'silverman': Silverman's rule (slightly more conservative than Scott's rule)
        - 'isj': Improved Sheather-Jones method (robust to multimodality, requires 50+ points)
        - 'cv': Cross-validation (computationally intensive but adaptive)
        If a float value is provided, it will be used directly as the bandwidth.
    grid_points : int, default=1000
        Number of points in the evaluation grid.
    grid_min : float, optional
        Minimum value for the evaluation grid. If None, uses min(data)
    grid_max : float, optional
        Maximum value for the evaluation grid. If None, uses max(data)
    mirror : tuple of (float or None, float or None), default=(None, None)
        Values about which to mirror the data at lower and upper bounds.
        - If mirror[0] is not None, data will be mirrored about this value at the lower bound,
          and any data points below this value will be removed.
        - If mirror[1] is not None, data will be mirrored about this value at the upper bound,
          and any data points above this value will be removed.
        Mirroring helps reduce edge effects in the KDE while enforcing bounds on the data distribution.
    
    Returns:
    -------
    grid : array
        The grid points where the KDE was evaluated.
    density : array
        The density estimate at each grid point.
    bw : float
        The bandwidth that was used.
    success : bool
        Whether the method was successful. False if fallback was used.
    method_used : str
        The method that was actually used (may differ from requested method if fallback occurred).
    """
    import numpy as np
    import warnings
    import logging
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    
    logger = logging.getLogger(__name__)
    
    # Ensure data is a numpy array and 1D
    data = np.asarray(data).flatten()
    
    # Need at least 2 points for KDE
    if len(data) < 2:
        raise ValueError("At least 2 data points are required for KDE")
    
    # Handle data mirroring for edge effects
    augmented_data = data.copy()
    mirror_lower, mirror_upper = mirror
    
    # Filter and mirror data if bounds are provided
    if mirror_lower is not None or mirror_upper is not None:
        # Filter data within bounds
        mask = np.ones(len(data), dtype=bool)
        if mirror_lower is not None:
            mask &= (data >= mirror_lower)
        if mirror_upper is not None:
            mask &= (data <= mirror_upper)
        
        augmented_data = data[mask].copy()
        
        if len(augmented_data) == 0:
            raise ValueError("All data points are outside the specified mirror bounds")
        
        # Calculate data range once
        data_range = np.max(augmented_data) - np.min(augmented_data)
        
        # Apply mirroring for lower bound
        if mirror_lower is not None:
            points_to_mirror = augmented_data[augmented_data < mirror_lower + data_range/2]
            if len(points_to_mirror) > 0:
                augmented_data = np.concatenate([2 * mirror_lower - points_to_mirror, augmented_data])
        
        # Apply mirroring for upper bound
        if mirror_upper is not None:
            points_to_mirror = augmented_data[augmented_data > mirror_upper - data_range/2]
            if len(points_to_mirror) > 0:
                augmented_data = np.concatenate([augmented_data, 2 * mirror_upper - points_to_mirror])
    
    # Calculate the standard deviation of the original data
    std_data = np.std(data)
    
    # Setup the evaluation grid based on the original data range
    if grid_min is None:
        grid_min = np.min(data) #- 3 * std_data
    if grid_max is None:
        grid_max = np.max(data) #+ 3 * std_data
    
    grid = np.linspace(grid_min, grid_max, grid_points)
    
    # Initialize bandwidth and success flag
    bw = None
    success = True
    method_original = method
    method_used = method  # Track the method actually used

    # if len aug data < 50, sample from original data until n=50
    if len(augmented_data) < 50:
        # Sample from original data
        n_needed = 50 - len(augmented_data)
        if n_needed > 0:
            # Sample with replacement
            sampled_data = np.random.choice(data, size=n_needed, replace=True)
            augmented_data = np.concatenate([augmented_data, sampled_data])
    
    # Check if method is a numeric bandwidth value
    if isinstance(method, (int, float)):
        bw = float(method)
        method = 'manual'  # Set method to 'manual' for tracking
        method_used = 'manual'
    else:
        # Calculate bandwidth based on the selected method
        # Note: We calculate bandwidth using augmented data for better edge handling
        if method == 'scott':
            # Scott's rule: h = n^(-1/5) * std
            n = len(augmented_data)
            bw = n ** (-1/5) * np.std(augmented_data)
        
        elif method == 'silverman':
            # Silverman's rule: h = (4/(3*n))^(1/5) * std
            n = len(augmented_data)
            bw = (4.0 / (3.0 * n)) ** (1/5) * np.std(augmented_data)
        
        elif method == 'isj':
            # Improved Sheather-Jones method
            if len(augmented_data) >= 50:

                try:
                    from KDEpy import FFTKDE
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Fit FFTKDE with ISJ bandwidth (but don't use it for evaluation)
                        # Just get the bandwidth value
                        kde_isj = FFTKDE(kernel='gaussian', bw='ISJ')
                        kde_isj.fit(augmented_data)
                        bw = kde_isj.bw
                        
                        # Now we have the ISJ bandwidth, but we'll use sklearn for evaluation
                        # This avoids the interpolation that causes zeros at the boundaries
                except Exception as e:
                    logger.exception(f"ISJ method failed. Falling back to Silverman's rule.")
                    method = 'silverman'
                    method_used = 'silverman'  # Update the method used
                    n = len(augmented_data)
                    bw = (4.0 / (3.0 * n)) ** (1/5) * np.std(augmented_data)
                    success = False
            else:
                logger.warning(f"Not enough data points for ISJ ({len(augmented_data)} available, 50 required). Falling back to Silverman's rule.")
                method = 'silverman'
                method_used = 'silverman'  # Update the method used
                n = len(augmented_data)
                bw = (4.0 / (3.0 * n)) ** (1/5) * np.std(augmented_data)
                success = False
        
        elif method == 'cv':
            # Cross-validation
            try:
                # Reshape data for sklearn (requires 2D array)
                data_2d = augmented_data.reshape(-1, 1)
                
                # Define the bandwidth parameter grid
                min_bw = 0.01
                max_bw = 1.0 * np.std(augmented_data)
                if min_bw > max_bw:
                    max_bw = 1  # Ensure max_bw is greater than min_bw

                param_grid = {'bandwidth': np.linspace(min_bw, max_bw, 15)}
                
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    KernelDensity(kernel='gaussian'),
                    param_grid,
                    cv=5  # 5-fold cross-validation
                )
                grid_search.fit(data_2d)
                
                # Get the best bandwidth
                bw = grid_search.best_params_['bandwidth']
            except Exception as e:
                logger.exception(f"Cross-validation failed. Falling back to Silverman's rule.")
                method = 'silverman'
                method_used = 'silverman'  # Update the method used
                n = len(augmented_data)
                bw = (4.0 / (3.0 * n)) ** (1/5) * np.std(augmented_data)
                success = False
        
        else:
            raise ValueError(f"Unknown method: {method}. Options are 'scott', 'silverman', 'isj', 'cv', or a numeric bandwidth value.")
    
    # Calculate the KDE using the computed bandwidth and augmented data
    try:
        # Always use sklearn for density estimation, regardless of the method used to calculate bandwidth
        # Reshape data and grid for sklearn
        augmented_data_2d = augmented_data.reshape(-1, 1)
        grid_2d = grid.reshape(-1, 1)
        
        # Fit KDE and evaluate
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(augmented_data_2d)
        log_density = kde.score_samples(grid_2d)
        density = np.exp(log_density)
    except Exception as e:
        logger.exception(f"Error calculating KDE: {str(e)}")
        grid = np.linspace(grid_min, grid_max, grid_points)
        density = np.zeros_like(grid)
        success = False
        method_used = 'failed'
    
    # Normalize the density so it integrates to 1
    # This is important especially when using mirroring
    if np.sum(density) > 0:
        dx = (grid_max - grid_min) / (grid_points - 1)
        density = density / (np.sum(density) * dx)
    
    return grid, density, bw, success, method_used