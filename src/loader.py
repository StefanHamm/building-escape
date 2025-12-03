import os
from typing import List

import numpy as np

def loadFloorPlan(path: str) -> np.ndarray:
    """
    Load a floor plan from a file where each character represents a cell.
    Reads the file line by line, handling potential whitespace.

    :param path: Path to the .fplan file.
    :return: Floor plan as a 2D numpy array of single characters (dtype='U1').
             Returns an empty 2D array (shape=(0,0)) if the file is empty,
             not found, or cannot be processed.
    """
    lines_data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace (including newline characters)
                stripped_line = line.strip()
                # Only process lines that are not empty after stripping
                if stripped_line:
                    # Convert the string line into a list of its individual characters
                    lines_data.append(list(stripped_line))

        if not lines_data:
            # Handle empty file or file with only whitespace
            print(f"Warning: Track file '{path}' is empty or contains only whitespace.")
            # Return an empty array with shape (0, 0)
            return np.array([[]], dtype='U1').reshape(0, 0)

        # Optional: Check for consistent line lengths (recommended for valid tracks)
        first_len = len(lines_data[0])
        if not all(len(row) == first_len for row in lines_data):
            # If lengths differ, NumPy will create an array with dtype=object,
            # which might cause issues later. It's often better to enforce consistency.
            print(f"Error: Floor Plan file '{path}' has inconsistent line lengths.")
            # Return an empty array or raise a ValueError
            # raise ValueError(f"Track file '{path}' has inconsistent line lengths.")
            return np.array([[]], dtype='U1').reshape(0, 0) # Returning empty for now

        # Convert the list of character lists into a NumPy array
        # 'U1' dtype ensures each element is treated as a single Unicode character
        matrixTrack = np.array(lines_data, dtype='U1')
        return matrixTrack

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Floorplan file not found at '{path}'") from e

    except Exception as e:
        raise Exception(f"Error loading fplan file '{path}': {e}") from e
    
    
def load_sff_from_npy(path: str) -> np.ndarray:
    """
    Load a static floor field (SFF) from a .npy file.

    :param path: Path to the .npy file.
    :return: SFF as a 2D numpy array of floats.
    """
    try:
        sff = np.load(path)
        return sff
    except FileNotFoundError as e:
        raise FileNotFoundError(f"SFF file not found at '{path}'") from e
    except Exception as e:
        raise Exception(f"Error loading SFF file '{path}': {e}") from e