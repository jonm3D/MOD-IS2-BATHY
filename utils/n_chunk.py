import numpy as np
import pandas as pd

def n_chunk(x0, y0, points_per_chunk, indices=None, overlap=0, max_gap_size=np.inf, additional_data=None):
    """
    Chunk data into segments based on x-coordinates, with options for overlap and gap detection.
    
    Parameters:
    -----------
    x0 : array-like
        X-coordinates of data points
    y0 : array-like
        Y-coordinates of data points
    points_per_chunk : int or float
        Number of points to include in each chunk (can be np.inf)
    indices : array-like, optional`
        Original indices of the data points (defaults to range(len(x0)))
    overlap : float, optional
        Fraction of overlap between consecutive chunks (0 to 1)
    max_gap_size : float, optional
        Maximum allowed gap in x values before splitting a chunk (default is np.inf)
    additional_data : dict, optional
        Dictionary of additional data arrays to include in chunks
        Keys are column names, values are arrays with same length as x0
    
    Returns:
    --------
    chunks : list of DataFrames
        List of DataFrames containing the chunked data
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
    
    # Create and sort the DataFrame
    df = pd.DataFrame(data).sort_values('x').reset_index(drop=True)
    
    chunks = []
    chunk_id = 0
    chunk_to_indices = {}
    
    # Keep track of the next starting index
    next_start_idx = 0

    while next_start_idx < len(df):
        # Get current chunk of data
        end_idx = min(next_start_idx + points_per_chunk, len(df))
        chunk_df = df.iloc[next_start_idx:end_idx].copy()
        
        # Calculate the next starting point based on overlap
        if np.isinf(points_per_chunk):
            # When points_per_chunk is inf, we want to exit the loop after this iteration
            step_size = len(df)
        else:
            step_size = max(1, int(points_per_chunk * (1 - overlap)))
        
        next_start_idx += step_size
        
        # If we have multiple points, check for gaps
        if len(chunk_df) > 1:
            # Calculate differences in x values
            x_diffs = np.diff(chunk_df['x'].values)
            gap_indices = np.where(x_diffs > max_gap_size)[0]
            
            if len(gap_indices) > 0:
                # We have gaps, split into subchunks
                start_idx = 0
                for gap_idx in gap_indices:
                    # Create a subchunk up to this gap
                    subchunk = chunk_df.iloc[start_idx:gap_idx+1].copy()
                    subchunk['chunk_id'] = chunk_id
                    chunks.append(subchunk)
                    chunk_to_indices[chunk_id] = subchunk['original_index'].tolist()
                    chunk_id += 1
                    start_idx = gap_idx + 1
                
                # Add the final subchunk after the last gap
                if start_idx < len(chunk_df):
                    final_subchunk = chunk_df.iloc[start_idx:].copy()
                    final_subchunk['chunk_id'] = chunk_id
                    chunks.append(final_subchunk)
                    chunk_to_indices[chunk_id] = final_subchunk['original_index'].tolist()
                    chunk_id += 1
                
                # Update next_start_idx to avoid duplicating points
                last_point_idx = chunk_df.index[-1]
                next_start_idx = df.index.get_loc(last_point_idx) + 1
                continue  # Skip the code below since we've processed this chunk
            
        # No gaps (or single point) - add as one chunk
        chunk_df['chunk_id'] = int(chunk_id)
        chunks.append(chunk_df)
        chunk_to_indices[chunk_id] = chunk_df['original_index'].tolist()
        chunk_id += 1
    
    return chunks


def mask_n_chunk(x0, y0, points_per_chunk, mask=None, indices=None, overlap=0.0, 
                 additional_data=None, max_gap_size=None):
    """
    Efficiently chunk data based on counting 'True' values in mask with specified overlap between chunks.
    Splits chunks into subchunks when x-value gaps exceed max_gap_size.
    
    Parameters:
    -----------
    x0 : array-like
        X-coordinates of data points
    y0 : array-like
        Y-coordinates of data points
    points_per_chunk : int
        Number of True values to include in each chunk
    mask : array-like, optional
        Boolean mask of same length as x0/y0. If None, all points are considered True.
    indices : array-like, optional
        Original indices of the data points (defaults to range(len(x0)))
    overlap : float, optional
        Fraction of overlap between consecutive chunks (0.0 to 0.99)
    additional_data : dict, optional
        Dictionary of additional data arrays to include in chunks
        Keys are column names, values are arrays with same length as x0
    max_gap_size : float, optional
        Maximum allowed gap between consecutive x values within a chunk.
        If a gap exceeds this value, the chunk is split into subchunks.
        
    Returns:
    --------
    chunks : list of DataFrames
        List of DataFrames containing the chunked data with chunk_id and subchunk_id columns
    """
    # Validate inputs
    if len(x0) != len(y0):
        raise ValueError("x0 and y0 must have the same length")
    
    # Initialize indices if not provided
    if indices is None:
        indices = np.arange(len(x0))
    
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(len(x0), dtype=bool)
        
    elif len(mask) != len(x0):
        raise ValueError("Mask must have the same length as x0/y0")
    
    # Create data dictionary with original indices
    data_dict = {'x': x0, 'y': y0, 'mask': mask, 'original_index': indices}
    
    # Add any additional data
    if additional_data is not None:
        for name, values in additional_data.items():
            if len(values) != len(x0):
                raise ValueError(f"Length of additional data '{name}' ({len(values)}) does not match length of x0 ({len(x0)})")
            data_dict[name] = values
    
    # Create and sort the DataFrame
    df = pd.DataFrame(data_dict).sort_values('x').reset_index(drop=True)
    
    # Validate overlap
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap fraction must be in range [0, 1)")
    
    # If no true values in mask or empty data, return a single chunk
    if len(df) == 0 or not any(df['mask']):
        df['chunk_id'] = 0
        df['subchunk_id'] = 0
        return [df]
    
    # Calculate stride (how many true values to advance between chunk starts)
    stride = max(1, round(points_per_chunk * (1 - overlap)))
    
    # Get all indices where mask is True
    true_indices = df.index[df['mask']].tolist()
    total_true = len(true_indices)
    
    # Handle the case where we don't have enough true values
    if total_true < points_per_chunk:
        df['chunk_id'] = 0
        df['subchunk_id'] = 0
        return [df.copy()]  # Return all data as a single chunk
    
    chunks = []
    chunk_id = 0
    
    # Process chunks using a uniform approach
    start_true_idx = 0
    
    while start_true_idx + points_per_chunk <= total_true:
        # Determine the end of this chunk (after points_per_chunk true values)
        end_true_idx = start_true_idx + points_per_chunk - 1
        
        # Find the data index right after the last true value in this chunk
        if end_true_idx + 1 < total_true:
            bin_end_idx = true_indices[end_true_idx + 1]
        else:
            bin_end_idx = len(df)
        
        # For the first chunk, start at index 0, otherwise at the true index
        bin_start_idx = 0 if start_true_idx == 0 else true_indices[start_true_idx]
        
        # Create the chunk
        current_chunk = df.iloc[bin_start_idx:bin_end_idx].copy()
        current_chunk['chunk_id'] = chunk_id
        
        # Split chunk into subchunks if max_gap_size is specified
        if max_gap_size is not None and len(current_chunk) > 1:
            subchunks = _split_chunk_by_gap(current_chunk, max_gap_size, chunk_id)
            chunks.extend(subchunks)
        else:
            current_chunk['subchunk_id'] = 0
            chunks.append(current_chunk)
        
        # Move to the next chunk start
        start_true_idx += stride
        chunk_id += 1
    
    # Handle any remaining true values
    if start_true_idx < total_true:
        bin_start_idx = true_indices[start_true_idx]
        current_chunk = df.iloc[bin_start_idx:].copy()
        current_chunk['chunk_id'] = chunk_id
        
        # Split chunk into subchunks if max_gap_size is specified
        if max_gap_size is not None and len(current_chunk) > 1:
            subchunks = _split_chunk_by_gap(current_chunk, max_gap_size, chunk_id)
            chunks.extend(subchunks)
        else:
            current_chunk['subchunk_id'] = 0
            chunks.append(current_chunk)
    
    return chunks

def _split_chunk_by_gap(chunk, max_gap_size, chunk_id):
    """
    Helper function to split a chunk into subchunks based on gaps in x values.
    """
    # Reset index to make slicing easier
    chunk = chunk.reset_index(drop=True)
    
    # Calculate gaps between consecutive x values
    x_diffs = chunk['x'].diff()
    
    # Find where gaps exceed max_gap_size (don't include the first index)
    split_indices = chunk.index[x_diffs > max_gap_size].tolist()
    
    # If no gaps exceed max_gap_size, return the original chunk
    if not split_indices:
        chunk['subchunk_id'] = 0
        return [chunk]
    
    # Split the chunk into subchunks
    subchunks = []
    subchunk_id = 0
    
    # Add start and end indices
    all_indices = [0] + split_indices + [len(chunk)]
    
    # Create subchunks
    for i in range(len(all_indices) - 1):
        start_idx = all_indices[i]
        end_idx = all_indices[i + 1]
        
        subchunk = chunk.iloc[start_idx:end_idx].copy()
        if len(subchunk) > 0:  # Only add non-empty subchunks
            subchunk['subchunk_id'] = subchunk_id
            subchunks.append(subchunk)
            subchunk_id += 1
    
    return subchunks