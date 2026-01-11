import numpy as np


def findExits(floorPlan: np.ndarray) -> list[tuple[int, int]]:
    """Finds all exits in the given floor plan.

    Args:
        floorPlan (np.ndarray): The floor plan as a 2D numpy array of single characters. 

    Returns:
        list[tuple[int, int]]: A list of tuples representing the coordinates of the exits.
    """
    exits = []
    rows, cols = floorPlan.shape
    for r in range(rows):
        for c in range(cols):
            if floorPlan[r, c] == 'E':  
                exits.append((r, c))
    return exits

def getAllWhiteCoords(floorPlan: np.ndarray) -> set[tuple[int, int]]:
    """Gets all coordinates of white (free) cells in the floor plan.

    Args:
        floorPlan (np.ndarray): The floor plan as a 2D numpy array of single characters. 

    Returns:
        set[tuple[int, int]]: A set of tuples representing the coordinates of all white cells.
    """
    white_coords = set()
    rows, cols = floorPlan.shape
    for r in range(rows):
        for c in range(cols):
            if floorPlan[r, c] == 'F':  
                white_coords.add((r, c))
    return white_coords

def getSafeWhiteCoords(floorPlan: np.ndarray, layoutSFF: np.ndarray ) -> set[tuple[int, int]]:
    # only return white coords that have sff<np.inf (not surrounded by walls)
    safe_white_coords = set()
    rows, cols = floorPlan.shape
    for r in range(rows):
        for c in range(cols):
            if floorPlan[r, c] == 'F' and layoutSFF[r, c] < np.inf:  
                safe_white_coords.add((r, c))
                
    return safe_white_coords