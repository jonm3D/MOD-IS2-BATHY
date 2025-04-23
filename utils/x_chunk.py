import numpy as np
import pandas as pd

def x_chunk(x0, y0, bin_size, indices=None, overlap=0, max_gap_size=np.inf, additional_data=None):
    """
    Chunks data based on specified bin size with support for overlap and gap detection.

    Parameters:
    ----------
    x0 : array-like
        x-values of the data
    y0 : array-like
        y-values of the data
    bin_size : float
        Size of each bin for chunking
    indices : array-like, optional
        Original indices of the data points. If None, uses array indices.
    overlap : float, optional
        Fraction of bin width to extend on each side. Default is 0.
    max_gap_size : float, optional
        Maximum allowed gap size in x-values; larger gaps split chunks. Default is np.inf.
    additional_data : dict, optional
        Dictionary of additional data arrays to include in chunks
        Keys are column names, values are arrays with same length as x0

    Returns:
    -------
    chunks : list of DataFrames
        Each DataFrame contains a chunk of the data with columns ['x', 'y', 'original_index', 'chunk_id']
        and any additional columns specified in additional_data
    """
    # Initialize indices if not provided
    if indices is None:
        indices = np.arange(len(x0))
    
    # Create base data dictionary
    data = {'x': x0, 'y': y0, 'original_index': indices}
    
    # Add any additional data
    if additional_data is not None:
        for name, values in additional_data.items():
            if len(values) != len(x0):
                raise ValueError(f"Length of additional data '{name}' ({len(values)}) does not match length of x0 ({len(x0)})")
            data[name] = values
    
    # Create and sort the data
    df = pd.DataFrame(data).sort_values('x').reset_index(drop=True)
    
    # Calculate bin edges based on data range and bin size
    min_x = df['x'].min()
    max_x = df['x'].max()
    
    # Calculate number of bins and create bin edges
    num_bins = max(1, int(np.ceil((max_x - min_x) / bin_size)))
    bin_edges = np.linspace(min_x, min_x + (num_bins * bin_size), num_bins + 1)
    
    chunks = []
    chunk_id = 0
    chunk_to_indices = {}
    
    # Process each bin
    for i in range(len(bin_edges) - 1):
        bin_width = bin_size  # bin width is now always bin_size
        
        # Calculate extended bin boundaries if overlap is specified
        lower_bound = bin_edges[i] - (overlap * bin_width)
        upper_bound = bin_edges[i + 1] + (overlap * bin_width)
        
        # Select points within this bin (including overlap)
        in_bin_mask = (df['x'] >= lower_bound) & (df['x'] < upper_bound)
        bin_df = df[in_bin_mask].copy()
        
        # Skip empty bins
        if len(bin_df) == 0:
            continue
            
        # Check for gaps if we have multiple points
        if len(bin_df) > 1 and not np.isinf(max_gap_size):
            # Calculate differences in x values
            x_diffs = np.diff(bin_df['x'].values)
            gap_indices = np.where(x_diffs > max_gap_size)[0]
            
            if len(gap_indices) > 0:
                # We have gaps, split into subchunks
                start_idx = 0
                for gap_idx in gap_indices:
                    # Create a subchunk up to this gap
                    subchunk = bin_df.iloc[start_idx:gap_idx+1].copy()
                    
                    subchunk['chunk_id'] = chunk_id
                    chunks.append(subchunk)
                    chunk_to_indices[chunk_id] = subchunk['original_index'].tolist()
                    chunk_id += 1
                        
                    start_idx = gap_idx + 1
                
                # Add the final subchunk after the last gap
                if start_idx < len(bin_df):
                    final_subchunk = bin_df.iloc[start_idx:].copy()
                    
                    final_subchunk['chunk_id'] = chunk_id
                    chunks.append(final_subchunk)
                    chunk_to_indices[chunk_id] = final_subchunk['original_index'].tolist()
                    chunk_id += 1
                
                # Continue to the next bin
                continue
        
        # No gaps (or gaps not checked) - add as one chunk
        bin_df['chunk_id'] = chunk_id
        chunks.append(bin_df)
        chunk_to_indices[chunk_id] = bin_df['original_index'].tolist()
        chunk_id += 1
    
    # If no valid chunks were found, return empty list
    if len(chunks) == 0:
        return []
    
    return chunks