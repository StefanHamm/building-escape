from dataclasses import dataclass
import numpy as np

MOORENEIGHBORHOOD_SIZE = 3


@dataclass
class Observation:
    mooreNeighborhoodSFF: np.ndarray  # 3x3

    def __post_init__(self):
        if self.mooreNeighborhoodSFF.shape != (MOORENEIGHBORHOOD_SIZE, MOORENEIGHBORHOOD_SIZE):
            raise ValueError(f"mooreNeighborhoodSFF must be 3x3, got {self.mooreNeighborhoodSFF.shape}")


@dataclass
class AgentState:
    x: int
    y: int
    done: bool = False


@dataclass
class AgentAction:
    move_direction: tuple[int, int]  # delta x, delta y
    # optional if current x and y coord


@dataclass
class Actions:
    UP: tuple = (0, -1)
    DOWN: tuple = (0, 1)
    LEFT: tuple = (-1, 0)
    RIGHT: tuple = (1, 0)
    UP_LEFT: tuple = (-1, -1)
    UP_RIGHT: tuple = (1, -1)
    DOWN_LEFT: tuple = (-1, 1)
    DOWN_RIGHT: tuple = (1, 1)

    # Map Moore neighborhood indices to actions
    MOORE_ACTIONS = [
        UP_LEFT,  # (0, 0)
        UP,  # (0, 1)
        UP_RIGHT,  # (0, 2)
        LEFT,  # (1, 0)
        None,  # (1, 1) - stay/invalid
        RIGHT,  # (1, 2)
        DOWN_LEFT,  # (2, 0)
        DOWN,  # (2, 1)
        DOWN_RIGHT  # (2, 2)
    ]
