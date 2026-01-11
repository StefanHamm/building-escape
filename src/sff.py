import numpy as np
from helper import findExits


# The function gets the MxN numpy array with the floor plan


def calculateSFF(floorPlan: np.ndarray, target_exits: list = None) -> np.array:
    """Calculates the static-floor-field (SFF) for a given floor plan. 
    The floorplan is a np.ndarray where each cell is represented by a character.

    Args:
        floorPlan (np.ndarray): The floor plan as a 2D numpy array of single characters. 

    Returns:
        np.ndarray: Returns the static-floor-field as a 2D numpy array of floats.
    """

    # get size of the floor plan
    rows, cols = floorPlan.shape

    # get the exits
    if target_exits is None:
        exits = findExits(floorPlan)
    else:
        exits = target_exits

    # intialize a matrix with inf values
    sff = np.full((rows, cols), np.inf)

    # for each exit set the sff value to 0
    for exit in exits:
        sff[exit] = 0.0

    previous_processed_coords = exits.copy()

    all_white_indices = set((r, c) for r in range(rows) for c in range(cols) if floorPlan[r, c] == 'F')
    # moore neighborhood 8 directions
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    processed_something = True

    while processed_something:
        processed_something = False

        new_state = sff.copy()
        # since we process one at a time we need to only check the neighbors of the previously processed coords
        new_processed_coords = []

        for r, c in previous_processed_coords:
            # get neighbor indices within bounds
            neighbors = [(r + dr, c + dc) for dr, dc in directions
                         if 0 <= r + dr < rows and 0 <= c + dc < cols]
            # check for white neighbors by being the all_white_indices set
            white_to_process = [(nr, nc) for nr, nc in neighbors if (nr, nc) in all_white_indices]

            for nr, nc in white_to_process:
                # those are the white cells that need processing
                # now get the new sff value for that cell based on its neighbors 
                # min(min(1/2+ direct neighbors),min(sqrt(2)/2 + diagonal neighbors))
                # its impossible to be out of bounds since a full black wall is arround
                direct_neighbors = [(nr + dr, nc + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                    if 0 <= nr + dr < rows and 0 <= nc + dc < cols]
                diagonal_neighbors = [(nr + dr, nc + dc) for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                                      if 0 <= nr + dr < rows and 0 <= nc + dc < cols]
                direct_min = min([sff[rn, cn] + 0.5 for rn, cn in direct_neighbors])
                diagonal_min = min([sff[rn, cn] + (np.sqrt(2) / 2) for rn, cn in diagonal_neighbors])
                new_value = min(direct_min, diagonal_min)
                if new_value < new_state[nr, nc]:
                    new_state[nr, nc] = new_value
                # add the new processed coord to the queue for the next iteration
                new_processed_coords.append((nr, nc))
                processed_something = True
        # remove the previous_processed_coords from all_white_indices set to avoid reprocessing
        all_white_indices.difference_update(previous_processed_coords)

        sff = new_state
        previous_processed_coords = list(set(new_processed_coords))

    return sff
