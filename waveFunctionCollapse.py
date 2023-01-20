import math
import ntpath
import queue
import sys
from typing import List, Tuple, Dict, Set, Any, Union
from moviepy.editor import ImageSequenceClip
import numpy as np
from PIL import Image as Img
import os
from matplotlib import pyplot as plt

USAGE_MSG = "Error: invalid input_examples. Usage: python3 wave_function_collapse.py <input_path> <pattern_size> " \
            "<out_height> <out_width> [flip] [rotate] [render_iterations] [render_video]"

# Output stdout messages for user
WAVE_COLLAPSED_MSG = "\nwave collapsed!"
FOUND_CONTRADICTION_MSG = "\nfound contradiction"

# Render while running variables
NUM_OF_ITERATIONS_BETWEEN_RENDER = 15  # The number of iterations between rendering in runtime
SHOW_PATTERNS = False  # Set to True to render all patterns and their probabilities
SAVE_PATTERNS = False  # Set to True to save all patterns to file
PRINT_RULES = False  # Set to True to print out all adjacency rules

# Video of runtime parameters
DEFAULT_FPS = 30  # Fps of the output video
DEFAULT_VIDEO_LENGTH = 6  # Length of the output video
DEFAULT_OUT_VID_HEIGHT = 1000  # Vertical size (in pixels) of the output video, which will preserve the original aspect ratio

# Status constants
CONTRADICTION = -2
WAVE_COLLAPSED = -1
RUNNING = 0


### Initialization functions ###


def initialize(input_path: str,
               pattern_size: int,
               out_height: int,
               out_width: int,
               flip: bool,
               rotate: bool) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int], np.ndarray, List[Dict[Tuple[int, int],
                                                                                                          Set[int]]]]:
    """
    Preforms the initialization phase of the wfc algorithm, namely - aggregate the patterns from the input_examples image,
    calculate their frequencies and initiate the coefficient_matrix
    :param input_path: The path to the input_examples image
    :param pattern_size: The size (width and height) of the patterns from the input_examples image, should be as small as
    possible for efficiency, but large enough to catch key features in the input_examples image
    :param out_height: The height of the output image
    :param out_width: The width of the output image
    :param flip: Set to True to calculate all possible flips of pattern as additional patterns
    :param rotate: Set to True to calculate all possible rotation of pattern as additional patterns
    :return: A tuple of:
        1. The wave matrix
        2. A list of all possible offsets for the patterns
        3. A list of all the frequencies of all the patterns
        4. A ndarray of the patterns
        5. A list of all the rules of adjacency for every pattern
    """
    # Get all possible offset directions for patterns
    directions = get_dirs(pattern_size)

    # Get a dictionary mapping patterns to their respective frequencies, then separate them
    pattern_to_freq = generate_patterns_and_frequencies(input_path, pattern_size, flip, rotate)
    patterns, frequencies = np.array(np.array([to_ndarray(tup) for tup in pattern_to_freq.keys()])), list(
        pattern_to_freq.values())
    if SHOW_PATTERNS:
        show_patterns(patterns, frequencies)
    if SAVE_PATTERNS:
        save_patterns(patterns, frequencies)

    # get all the rules of adjacency for all patterns
    rules = get_rules(patterns, directions)

    if PRINT_RULES:
        print_adjacency_rules(rules)

    # init the coefficient_matrix, representing  the wave function
    coefficient_matrix = np.full((out_height, out_width, len(patterns)), True, dtype=bool)
    return coefficient_matrix, directions, frequencies, patterns, rules


def generate_patterns_and_frequencies(path: str, N: int, flip: bool = True, rotate: bool = True) -> Dict[tuple[Any, ...], int]:
    """
    Extracts N by N subimages from an image from a given path and returns a dictionary of patterns to their frequency.
    Optionally includes flipped and rotated versions of the subimages.

    :param path: A string containing the path to the image file
    :param N: An integer specifying the size of the subimages
    :param flip: A boolean indicating whether to include flipped versions of the subimages (defaults to True)
    :param rotate: A boolean indicating whether to include rotated versions of the subimages (defaults to True)
    :return: A tuple with:
     1. The input_examples image as numpy array.
     2. A dictionary mapping each pattern as a tuple to its respective frequency
    """
    # Open the image using PIL and convert the image to a numpy array
    im = np.array(Img.open(path))

    # Check if the array has more than 3 channels
    if im.shape[2] > 3:
        # If the array has more than 3 channels, reduce the number of channels to 3
        im = im[:, :, :3]

    patterns = get_patterns_from_image(im, N, flip, rotate)

    patterns = [to_tuple(pattern) for pattern in patterns]
    pattern_to_freq = {item: patterns.count(item) for item in patterns}
    return pattern_to_freq


def get_patterns_from_image(im: np.ndarray, N: int, flip: bool = True, rotate: bool = True) -> List[np.ndarray]:
    """
    Extracts N by N subimages from an image and returns a dictionary of patterns to their frequency.
    Optionally includes flipped and rotated versions of the subimages.

    :param im: The image from which to extract the patterns, as a numpy array
    :param N: An integer specifying the size of the subimages
    :param flip: A boolean indicating whether to include flipped versions of the subimages (defaults to True)
    :param rotate: A boolean indicating whether to include rotated versions of the subimages (defaults to True)
    :return: A list of all the patterns of size N*N inside the input_examples image
    """
    # Generate a list of indices for the rows and columns of the image
    row_indices = np.arange(im.shape[0] - N + 1)
    col_indices = np.arange(im.shape[1] - N + 1)
    # Reshape the array of tiles into a list
    patterns = []
    for i in row_indices:
        for j in col_indices:
            patterns.append(im[i:i + N, j:j + N, :])
    # Optionally include flipped and rotated versions of the tiles
    flipped, rotated = [], []
    if flip:
        flipped = [np.flip(pattern, axis=axis) for axis in [0, 1] for pattern in patterns]
    if rotate:
        rotated = [np.rot90(pattern, k=k) for k in range(1, 4) for pattern in patterns]
    if flip:
        patterns += flipped
    if rotate:
        patterns += rotated
    return patterns


def to_tuple(array: np.ndarray) -> Union[Tuple, np.ndarray]:
    """
    Convert array to tuple.

    :param array: The array to be converted. Can be a NumPy ndarray or a nested sequence.
    :return: The input_examples array as a tuple with the same structure.
    """
    if isinstance(array, np.ndarray):
        return tuple(map(to_tuple, array))
    else:
        return array


def to_ndarray(tup: Tuple) -> Union[Tuple, np.ndarray]:
    """
    Convert tuple to NumPy ndarray.

    :param tup: The tuple to be converted. Can be a nested tuple.
    :return: The input_examples tuple as a NumPy ndarray with the same structure.
    """
    if isinstance(tup, tuple):
        return np.array(list(map(to_ndarray, tup)))
    else:
        return tup


def get_dirs(n: int) -> List[Tuple[int, int]]:
    """
    Get the coordinates around a pattern.
    This function returns a list of all coordinates around a pattern of size `n`, starting from the top left and ending at
    the bottom right. The center point (0, 0) is excluded from the list.

    :param n: The size of the pattern.
    :return: A list of coordinates around the pattern.
    """
    dirs = [(i, j) for j in range(-n + 1, n) for i in range(-n + 1, n)]
    dirs.remove((0, 0))
    return dirs


def mask_with_offset(pattern: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    """
    Get a subarray of a pattern based on an offset.
    This function returns a subarray of `pattern`, which is all entries that are inside the intersection of the `pattern`
    with another pattern offset by `offset`.

    :param pattern: an N*N*channels ndarray.
    :param offset: a 2D vector.
    :return: a subarray of `pattern` that intersects with another pattern by 'offset'.
    """
    x_offset, y_offset = offset
    if abs(x_offset) > len(pattern) or abs(y_offset) > len(pattern[0]):
        return np.array([[]])
    return pattern[max(0, x_offset):min(len(pattern) + x_offset,
                                        len(pattern)), max(0, y_offset):min(len(pattern[0]) + y_offset, len(pattern[0])), :]


def check_for_match(p1: np.ndarray, p2: np.ndarray, offset: Tuple[int, int]) -> bool:
    """
    checks whether 2 patterns with a given offset from each other match

    :param p1: first pattern
    :param p2: second pattern
    :param offset: offset of the first pattern
    :return: true if p2 equals to p1_ind with offset
    """
    p1_offset = mask_with_offset(p1, offset)
    p2_offset = mask_with_offset(p2, flip_dir(offset))
    return np.all(np.equal(p1_offset, p2_offset))


def flip_dir(d: Tuple[int, int]) -> Tuple[int, int]:
    """
    Flips the direction of the given 2D vector d

    :param d: A 2D vector
    :return: The input_examples vector multiplied by -1,-1
    """
    return -1 * d[0], -1 * d[1]


def get_rules(patterns: np.ndarray, directions: List[Tuple[int, int]]) -> List[Dict[Tuple[int, int], Set[int]]]:
    """
    Creates the rules data structure, which is a list where entry i holds a dictionary that maps offset (x,y) to a set of
    indices of all patterns matching there

    :param directions: An array of all surrounding possible offsets
    :param patterns: The list of all the patterns
    :return:The rules list
    """
    rules = [{dire: set() for dire in directions} for _ in range(len(patterns))]
    for i in range(len(patterns)):
        for d in directions:
            for j in range(i, len(patterns)):
                if check_for_match(patterns[i], patterns[j], d):
                    rules[i][d].add(j)
                    rules[j][flip_dir(d)].add(i)
    return rules


### WaveFunctionCollapse iteration functions ###

def get_min_entropy_coordinates(coefficient_matrix: np.ndarray, frequencies: List[int]) -> Tuple[int, int, int]:
    """
    Calculate the coordinates of the cell with the minimum entropy, and returns the row, column and the state of the wave (
    collapsed/running)

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
    :param rules: A list of dictionaries, where each dictionary represents the possible patterns that can be propagated in a
    specific direction for a specific pattern in the original cell.
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
    This function preforms the propagation part of the wfc algorithm. after collapsing one cell in 'observe', this function
    will propagate this change to all relevant cells.

    :param min_entropy_pos: A tuple of integers representing the position of the cell that was collapsed in 'observe'
    :param coefficient_matrix: A numpy array representing the matrix of coefficients of the entire wave
    :param rules: A list of dictionaries, where each dictionary represents a pattern, and contains a mapping of directions to a
    set of possible patterns in the direction from the pattern of the dict
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


def wave_function_collapse(input_path: str, pattern_size: int, out_height: int, out_width: int, flip: bool,
                           rotate: bool, render_iterations: bool, render_video: bool) -> np.ndarray:
    """
    The main function of the program, will preform the wave function collapse algorithm.
    Given an input_examples image, the function will randomly generate an output image of any size, where each pixel in the
    output image resembles a small, local environment in the input_examples image.

    :param flip: Set to True to calculate all possible flips of pattern as additional patterns
    :param rotate: Set to True to calculate all possible rotation of pattern as additional patterns
    :param input_path: The path of the input_examples image of the algorithm, from which to extract the patterns
    :param pattern_size: The size (width and height) of the patterns from the input_examples image, should be as small as
    possible for efficiency, but large enough to catch key features in the input_examples image
    :param out_height: The height of the output image
    :param out_width: The width of the output image
    :param render_iterations:Set to True to render images in runtime every NUM_OF_ITERATIONS_BETWEEN_RENDER iteratoins
    :param render_video: Set to True to render video of the run of the algorithm
    :return: A numpy array representing the output image
    """
    # Get the initial coefficient_matrix, patterns, frequencies, rules and directions of offsets
    coefficient_matrix, directions, frequencies, patterns, rules = initialize(input_path, pattern_size, out_height,
                                                                              out_width, flip, rotate)

    # Initialize control parameters
    status = 1
    iteration = 0

    # If render_video, initialize the images list
    if render_video:
        images = [image_from_coefficients(coefficient_matrix, patterns)[1]]

    # Iterate over the steps of the algorithm: observe, collapse, propagate until the wave collapses
    while status != WAVE_COLLAPSED:
        iteration += 1
        # Observe and collapse
        min_entropy_pos, coefficient_matrix, status = observe(coefficient_matrix, frequencies)

        if status == CONTRADICTION:
            print(FOUND_CONTRADICTION_MSG)
            exit(-1)

        # Get current progress status
        collapsed, image = image_from_coefficients(coefficient_matrix, patterns)

        # Update the progress bar
        progress_bar(out_height * out_width, collapsed)

        # If render_video, add current iteration to the images list
        if render_video and not status == WAVE_COLLAPSED:
            images.append(image)

        # If the wave collapsed, stop the iterations
        if status == WAVE_COLLAPSED:
            print(WAVE_COLLAPSED_MSG)
            show_iteration(iteration, patterns, coefficient_matrix)

            # If render_video, save the image list to a video
            if render_video:
                images.append(image)
                save_iterations_to_video(images, input_path)

            # Save the output image and return it
            return save_collapsed_wave(coefficient_matrix, input_path, patterns)

        # If render_iterations and NUM_OF_ITERATIONS_BETWEEN_RENDER passed from last render, render this iteration
        if render_iterations and iteration % NUM_OF_ITERATIONS_BETWEEN_RENDER == 0:
            show_iteration(iteration, patterns, coefficient_matrix)

        # Propagate
        coefficient_matrix = propagate(min_entropy_pos, coefficient_matrix, rules, directions)


### Rendering Functions ###

def save_collapsed_wave(coefficient_matrix: np.ndarray, input_path: str, patterns: np.ndarray) -> np.ndarray:
    """
    Saves an image of the collapsed wave, with width 1000 and preserves the aspect ratio
    :param coefficient_matrix: The wave matrix
    :param input_path: The path of the input_examples image
    :param patterns: The numpy array of the patterns
    :return: An image of the collapsed wave
    """
    # Get the dimensions of the output image
    w, h, _ = coefficient_matrix.shape
    num_channels = patterns[0].ndim

    # Create the output image as an array
    final_image = patterns[np.where(coefficient_matrix[:, :])[2]][:, 0, 0, :].reshape(w, h, num_channels)

    # Calculate the upscale_parameter
    upscale_parameter = (DEFAULT_OUT_VID_HEIGHT, (min(w, h) * DEFAULT_OUT_VID_HEIGHT) // max(w, h))

    # Create the image from the array and up sample it
    im = Img.fromarray(final_image).resize(upscale_parameter, resample=Img.NONE)

    # Save the image
    file_name = f"WFC_{ntpath.basename(input_path)}"
    im.save(file_name)
    return final_image


def show_iteration(iteration: int, patterns: np.ndarray, coefficient_matrix: np.ndarray) -> np.ndarray:
    """
    Shows the state of the wave in this iteration of the algorithm
    :param iteration: The iteration of the algorithm
    :param patterns: The ndarray of the patterns
    :param coefficient_matrix: The ndarray representing the wave
    :return: A ndarray representing the image of the wave in the current iterations, all un-collapsed cells are with the
    mean color of all the valid patterns for the cell.
    """
    collapsed, res = image_from_coefficients(coefficient_matrix, patterns)
    w, h, _ = res.shape
    fig, axs = plt.subplots()
    axs.imshow(res)
    axs.set_title(f"cells collapsed: {collapsed} out of {w * h}, done {round(100 * collapsed / (w * h), 2)}%")
    fig.suptitle(f"iteration number: {iteration}")
    plt.show()
    return res


def save_iterations_to_video(images: List[np.ndarray], input_path: str) -> None:
    """
    Saves all the images of iterations of the algorithm to a video
    :param images: A list of ndarrays of the state of the wave during iterations of the algorithm
    :param input_path: The path of the input_examples image
    """
    w, h, _ = images[0].shape

    # Calculate the upscale_parameter for the video
    upscale_parameter = DEFAULT_OUT_VID_HEIGHT // max(w, h)

    # Calculate the time_sample_parameter for the video, so that the output video will be in DEFAULT_FPS fps
    time_sample_parameter = 1
    if len(images) > DEFAULT_FPS * DEFAULT_VIDEO_LENGTH:
        time_sample_parameter = len(images) // (DEFAULT_FPS * DEFAULT_VIDEO_LENGTH)

    # Upscale and subsample over time the images list
    images = np.array(images)
    images = np.kron(images[::time_sample_parameter, :, :, :], np.ones((upscale_parameter, upscale_parameter, 1)))
    images = [images[i] for i in range(images.shape[0])]

    # Save the output video
    out_name = f"WFC_{ntpath.basename(input_path).split('.')[0]}.mp4"
    clip = ImageSequenceClip(images, fps=DEFAULT_FPS)
    clip.write_videofile(out_name, fps=DEFAULT_FPS)


def image_from_coefficients(coefficient_matrix: np.ndarray, patterns: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Generates an image of the state of the wave function mid-run. every cell is the mean of all valid patterns for it
    :param coefficient_matrix: The ndarray of the wave function
    :param patterns: The ndarray of the patterns
    :return: A tuple:
        1. The number of collapsed cells in the coefficient_matrix
        2. The image of the state of the wave function
    """
    # todo find a more efficient way to do this
    r, c, num_patterns = coefficient_matrix.shape

    # Create the resulting image of the wave function
    res = np.empty((r, c, 3))

    # Iterate over all cells of coefficient_matrix
    for row in range(r):
        for col in range(c):
            # For each cell, find the valid patterns in it
            valid_patterns = np.where(coefficient_matrix[row, col])[0]

            # Assign the corresponding cell in the output to be the mean of all valid patterns in the cell
            res[row, col] = np.mean(patterns[valid_patterns], axis=0)[0, 0]

    # Return the number of collapsed cells and the result image
    return np.count_nonzero(np.sum(coefficient_matrix, axis=2) == 1), res.astype(int)


def print_adjacency_rules(rules: List[Dict[Tuple[int, int], Set[int]]]) -> None:
    """
    Prints all adjacency rules
    :param rules: The list of rules per direction per pattern per other pattern
    """
    for i in range(len(rules)):
        print(f"pattern number {i}:")
        for k, v in rules[i].items():
            print(f"key: {k}, values: {v}")


def progress_bar(max_work: int, curr_work: int) -> None:
    """
    Prints a progress bar of the algorithm run to stdout
    :param max_work: Total work to be done, in this case - number of cells to collapse
    :param curr_work: The amount of work done so far, in this case - number of cells collapsed
    """
    # Calculate percentage of work done
    percentage = int(100 * (curr_work / float(max_work)))

    # Create and print the progress bar
    bar = '█' * percentage + '-' * (100 - percentage)
    print(f"\r|{bar}| {round(percentage, 2)}% ", end='\b')


def save_patterns(patterns: np.ndarray, freq: List[int], output_path: str = 'output') -> None:
    """
    Saves a list of image tiles to new image files in a given output path.

    :param patterns: A list of image tiles, each represented as a numpy array.
    :param freq: A list with the number of occurrences of pattern i in the i'th place.
    :param output_path: A string containing the path to the output directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sum_of_freq = sum(freq)
    for i in range(len(patterns)):
        fig, axs = plt.subplots()
        axs.imshow(patterns[i])
        axs.set_title(f"pattern No. {i + 1}")
        axs.set_xticks([])
        axs.set_yticks([])
        fig.suptitle(f"Number of occurrences: {freq[i]}, probability: {round(freq[i] / sum_of_freq, 2)}")
        plt.savefig(os.path.join(output_path, f'pattern_{i}.jpg'))


def show_patterns(patterns: np.ndarray, freq: List[int]) -> None:
    """
    Display the patterns in a grid

    :param patterns: A numpy array of patterns
    :param freq: A list of integers representing the frequency of each pattern
    """
    plt.figure(figsize=(10, 10))
    freq_sum = sum(freq)
    for m in range(len(patterns)):
        axs = plt.subplot(int(math.sqrt(len(patterns))), math.ceil(len(patterns) / int(math.sqrt(len(patterns)))), m + 1)
        axs.imshow(patterns[m])
        axs.set_xticks([])
        axs.set_yticks([])
        plt.title(f"num of appearances: {freq[m]}\n prob: {round(freq[m] / freq_sum, 2)}")
    plt.show()


if __name__ == "__main__":
    try:
        if len(sys.argv) > 9:
            raise IndexError
        input_path, pattern_size, out_height, out_width = sys.argv[1:5]
        pattern_size, out_height, out_width = int(pattern_size), int(out_height), int(out_width)
        flip = sys.argv[5] if len(sys.argv) >= 6 else False
        rotate = sys.argv[6] if len(sys.argv) >= 7 else False
        render_iterations = sys.argv[7] if len(sys.argv) >= 8 else True
        render_video = sys.argv[8] if len(sys.argv) == 9 else True
    except (TypeError, ValueError, IndexError):
        print(USAGE_MSG)
    else:
        wave_function_collapse(input_path, pattern_size, out_height, out_width, flip, rotate, render_iterations, render_video)
