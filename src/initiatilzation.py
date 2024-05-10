import math
import os
from typing import Tuple, List, Dict, Set, Any, Union

import numpy as np
from PIL import Image as Img
from matplotlib import pyplot as plt


def initialize(input_path: str,
               pattern_size: int,
               out_height: int,
               out_width: int,
               flip: bool,
               rotate: bool) -> Tuple[
    np.ndarray, List[Tuple[int, int]], List[int], np.ndarray, List[Dict[Tuple[int, int], Set[int]]]]:
    """
    Preforms the initialization phase of the wfc algorithm, namely - aggregate the patterns from the
    input_examples image, calculate their frequencies and initiate the coefficient_matrix
    :param input_path: The path to the input_examples image
    :param pattern_size: The size (width and height) of the patterns from the input_examples image, should be as
    small as possible for efficiency, but large enough to catch key features in the input_examples image
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


def generate_patterns_and_frequencies(path: str, N: int, flip: bool = True, rotate: bool = True) -> Dict[
    tuple[Any, ...], int]:
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
    This function returns a list of all coordinates around a pattern of size `n`, starting from the top left and
    ending at the bottom right. The center point (0, 0) is excluded from the list.

    :param n: The size of the pattern.
    :return: A list of coordinates around the pattern.
    """
    dirs = [(i, j) for j in range(-n + 1, n) for i in range(-n + 1, n)]
    dirs.remove((0, 0))
    return dirs


def get_rules(patterns: np.ndarray, directions: List[Tuple[int, int]]) -> List[Dict[Tuple[int, int], Set[int]]]:
    """
    Creates the rules data structure, which is a list where entry i holds a dictionary that maps offset (x,y) to a
    set of indices of all patterns matching there

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


def mask_with_offset(pattern: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    """
    Get a subarray of a pattern based on an offset.
    This function returns a subarray of `pattern`, which is all entries that are inside the intersection of the
    `pattern` with another pattern offset by `offset`.

    :param pattern: an N*N*channels ndarray.
    :param offset: a 2D vector.
    :return: a subarray of `pattern` that intersects with another pattern by 'offset'.
    """
    x_offset, y_offset = offset
    if abs(x_offset) > len(pattern) or abs(y_offset) > len(pattern[0]):
        return np.array([[]])
    return pattern[max(0, x_offset):min(len(pattern) + x_offset,
                                        len(pattern)),
           max(0, y_offset):min(len(pattern[0]) + y_offset, len(pattern[0])), :]


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


SHOW_PATTERNS = False  # Set to True to render all patterns and their probabilities
SAVE_PATTERNS = False  # Set to True to save all patterns to file
PRINT_RULES = False  # Set to True to print out all adjacency rules


def print_adjacency_rules(rules: List[Dict[Tuple[int, int], Set[int]]]) -> None:
    """
    Prints all adjacency rules
    :param rules: The list of rules per direction per pattern per other pattern
    """
    for i in range(len(rules)):
        print(f"pattern number {i}:")
        for k, v in rules[i].items():
            print(f"key: {k}, values: {v}")


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
        axs = plt.subplot(int(math.sqrt(len(patterns))), math.ceil(len(patterns) / int(math.sqrt(len(patterns)))),
                          m + 1)
        axs.imshow(patterns[m])
        axs.set_xticks([])
        axs.set_yticks([])
        plt.title(f"num of appearances: {freq[m]}\n prob: {round(freq[m] / freq_sum, 2)}")
    plt.show()
