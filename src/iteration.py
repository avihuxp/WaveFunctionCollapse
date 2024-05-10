import queue
from typing import List, Tuple, Dict, Set

import numpy as np

CONTRADICTION = -2
WAVE_COLLAPSED = -1
RUNNING = 0


def get_min_entropy_coordinates(coefficient_matrix: np.ndarray, frequencies: List[int]) -> Tuple[int, int, int]:
    """
    Calculate the coordinates of the cell with the minimum entropy, and returns the row, column and the state of
    the wave (collapsed/running)

    :param coefficient_matrix: A matrix of coefficients representing the wave state
    :param frequencies: List of frequencies of patterns in the wave
    :return: Tuple of integers: row, column, state of wave (collapsed/running)
    """
    # Calculate the probability of each pattern
    prob = np.array(frequencies) / np.sum(frequencies)

    # Calculate the entropies for each cell
    entropies = np.sum(coefficient_matrix.astype(int) * prob, axis=2)

    # Set entropy to 0 for all collapsed cells
    entropies[np.sum(coefficient_matrix, axis=2) == 1] = 0

    if np.sum(entropies) == 0:
        return -1, -1, WAVE_COLLAPSED

    # Get all indices of minimal non-zero cells
    min_indices = np.argwhere(entropies == np.amin(entropies, initial=np.max(entropies), where=entropies > 0))

    # Check if entropy of all cells is 0, i.e. the wave is fully collapsed
    if min_indices.shape[0] == 0:
        return -1, -1, WAVE_COLLAPSED

    # Choose a random index from the list of minimum indices
    min_index = min_indices[np.random.randint(0, min_indices.shape[0])]
    return min_index[0], min_index[1], RUNNING


def is_cell_collapsed(coefficient_matrix: np.ndarray, cell_pos: Tuple[int, int]) -> bool:
    """
    Check if the cell located at `cell_pos` in the `coefficient_matrix` is collapsed.

    :param coefficient_matrix:  A matrix of coefficients representing the wave state.
    :param cell_pos: Tuple of integers representing the position of the cell in the matrix (x, y).
    :return: A boolean indicating whether the cell is collapsed (True) or not (False).
    """
    return np.sum(coefficient_matrix[cell_pos[0], cell_pos[1], :]) == 1


def propagate_cell(orig_cell: Tuple[int, int], direction: Tuple[int, int], coefficient_matrix: np.ndarray,
                   rules: List[Dict[Tuple[int, int], Set[int]]]) -> np.ndarray:
    """
    Propagates patterns restrictions from one cell to an adjacent cell in a given direction according to a set of rules.

    :param orig_cell: The position of the original cell (x, y) in the coefficient matrix
    :param direction: The direction to propagate patterns in (dx, dy)
    :param coefficient_matrix: A matrix of coefficients representing the possible patterns in each cell
    :param rules: A list of dictionaries, where each dictionary represents the possible patterns that can be
    propagated in a specific direction for a specific pattern in the original cell.
    :return: The propagated patterns for the adjacent cell, in the form of a one cell in the matrix of coefficients
    """
    # Get the coordinates of the cell to which we will propagate
    adjacent_cell_pos = orig_cell[0] + direction[0], orig_cell[1] + direction[1]

    # Get all valid patterns of the cell to which we will propagate
    valid_patterns_in_adjacent_cell = coefficient_matrix[adjacent_cell_pos]

    # Get the patterns that are valid in the target cell, by index
    possible_patterns_in_orig_cell = np.where(coefficient_matrix[orig_cell])[0]

    # Create a vector full of False with the same shape as the target cell
    possibilities_in_dir = np.zeros(coefficient_matrix[orig_cell].shape, bool)

    # Accumulate all possible patterns in the direction
    for pattern in possible_patterns_in_orig_cell:
        possibilities_in_dir[list(rules[pattern][direction])] = True

    # Multiply the target cell by possible patterns form original cell
    return np.multiply(possibilities_in_dir, valid_patterns_in_adjacent_cell)


def in_matrix(pos: Tuple[int, int], dims: Tuple[int, ...]) -> bool:
    """
    :param pos: A tuple of integers representing the position of an element in a matrix
    :param dims: A tuple of integers representing the dimensions of the matrix
    :return: A boolean indicating whether the given position is within the bounds of the matrix
    """
    return 0 <= pos[0] < dims[0] and 0 <= pos[1] < dims[1]


def propagate(min_entropy_pos: Tuple[int, int],
              coefficient_matrix: np.ndarray,
              rules: List[Dict[Tuple[int, int], Set[int]]],
              directions: List[Tuple[int, int]]) -> np.ndarray:
    """
    This function preforms the propagation part of the wfc algorithm. after collapsing one cell in 'observe',
    this function will propagate this change to all relevant cells.

    :param min_entropy_pos: A tuple of integers representing the position of the cell that was collapsed in 'observe'
    :param coefficient_matrix: A numpy array representing the matrix of coefficients of the entire wave
    :param rules: A list of dictionaries, where each dictionary represents a pattern, and contains a mapping
    of directions to a set of possible patterns in the direction from the pattern of the dict
    :param directions: A list of tuples of integers representing the directions in which the cell is being propagated
    :return: the updated coefficient matrix, after propagation
    """
    # create a queue of positions of cells to update
    propagation_queue = queue.Queue()
    propagation_queue.put(min_entropy_pos)

    # Loop until no more cells are left to be updated
    while not propagation_queue.empty():
        cell = propagation_queue.get()

        # For each direction of the current cell
        for direction in directions:
            adjacent_cell_pos = cell[0] + direction[0], cell[1] + direction[1]

            # If the adjacent cell is not collapsed, propagate to it
            if in_matrix(adjacent_cell_pos, coefficient_matrix.shape) and not \
                    is_cell_collapsed(coefficient_matrix, adjacent_cell_pos):
                updated_cell = propagate_cell(cell, direction, coefficient_matrix, rules)

                # If the propagation to the adjacent cell changed its wave, update the coefficient matrix
                if not np.array_equal(coefficient_matrix[adjacent_cell_pos], updated_cell):
                    coefficient_matrix[adjacent_cell_pos] = updated_cell

                    # if the adjacent cell is not in the propagation_queue, add it
                    if adjacent_cell_pos not in propagation_queue.queue:
                        propagation_queue.put(adjacent_cell_pos)
    return coefficient_matrix


def observe(coefficient_matrix: np.ndarray, frequencies: List[int]) -> Tuple[Tuple[int, int], np.ndarray, int]:
    """
    The function preforms the 'observe' phase oof the wfc algorithm. it searches for the cell with the minimal entropy,
    and collapses it, based on possible patterns in the cell and there respective frequencies.

    :param coefficient_matrix: A numpy array representing the matrix of the wave
    :param frequencies: A list of integers representing the frequency of each pattern withing the input_examples image
    :return: A tuple containing:
     1. A tuple of integers representing the position of the cell with the lowest entropy.
     2. An updated numpy array of the wave after the collapse.
     3. An integer representing the status of the wave function.
    """
    # If contradiction
    if np.any(~np.any(coefficient_matrix, axis=2)):
        return (-1, -1), coefficient_matrix, CONTRADICTION
    # Get min pos
    min_entropy_pos_x, min_entropy_pos_y, status = get_min_entropy_coordinates(coefficient_matrix, frequencies)
    min_entropy_pos = (min_entropy_pos_x, min_entropy_pos_y)
    # If fully collapsed
    if status == WAVE_COLLAPSED:
        return (min_entropy_pos_x, min_entropy_pos_y), coefficient_matrix, WAVE_COLLAPSED
    # Collapse the cell at min_entropy_pos
    coefficient_matrix = collapse_single_cell(coefficient_matrix, frequencies, min_entropy_pos)
    return min_entropy_pos, coefficient_matrix, RUNNING


def collapse_single_cell(coefficient_matrix: np.ndarray, frequencies: List[int],
                         min_entropy_pos: Tuple[int, int]) -> np.ndarray:
    """
    Collapses a single cell at min_entropy_pos to a single pattern, randomly weighted by the frequencies.

    :param coefficient_matrix: A numpy array representing the matrix of the wave
    :param frequencies: A list of integers representing the frequency of each pattern withing the input_examples image
    :param min_entropy_pos: the position of the cell to collapse
    :return: the update matrix of the wave, with the cell collapsed
    """
    # Get indices of optional patterns at min_entropy_pos
    relevant_ind = np.where(coefficient_matrix[min_entropy_pos])[0]

    # Get frequencies for relevant patterns at min_entropy_pos
    relevant_freq = np.array(frequencies)[relevant_ind]

    # Collapse cell to a pattern randomly, weighted by the frequencies
    chosen_pattern_ind = np.random.choice(relevant_ind, p=relevant_freq / np.sum(relevant_freq))

    # Set possibility of all patterns other than the chosen pattern to False
    coefficient_matrix[min_entropy_pos] = np.full(coefficient_matrix[min_entropy_pos].shape[0], False, dtype=bool)
    coefficient_matrix[min_entropy_pos[0], min_entropy_pos[1], chosen_pattern_ind] = True
    return coefficient_matrix
