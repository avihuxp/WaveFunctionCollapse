import math
import queue
from moviepy.editor import ImageSequenceClip
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

CONTRADICTION = -2
WAVE_COLLAPSED = -1
RUNNING = 0
SAVE_VIDEO = True


def get_patterns(path, N, flip=True, rotate=True):
    """
    Extracts N by N subimages from an image and returns a list of the subimages.
    Optionally includes flipped and rotated versions of the subimages.
    :param path: a string containing the path to the image file
    :param N: an integer specifying the size of the subimages
    :param flip: a boolean indicating whether to include flipped versions of the subimages (defaults to True)
    :param rotate: a boolean indicating whether to include rotated versions of the subimages (defaults to True)
    :return: A tuple with:
     1. The input image as numpy array.
     2. A list of the N by N subimages of the image, with optional flipped and rotated versions.
    """
    # Open the image using PIL and convert the image to a numpy array
    im = np.array(Image.open(path))

    # Get the dimensions of the image
    image_height, image_width, channels = im.shape

    # Check if the array has more than 3 channels
    if channels > 3:
        # If the array has more than 3 channels, reduce the number of channels to 3
        im = im[:, :, :3]
        channels = 3

    # Generate a list of indices for the rows and columns of the image
    row_indices = np.arange(image_height - N + 1)
    col_indices = np.arange(image_width - N + 1)

    # Reshape the array of tiles into a list
    patterns = []

    for i in row_indices:
        for j in col_indices:
            patterns.append(im[i:i + N, j:j + N, :])

    # Optionally include flipped and rotated versions of the tiles
    flipped, rotated = [], []
    if flip:
        flipped = [np.flip(pattern, axis=axis) for axis in [0, 1, -1] for pattern in patterns]
    if rotate:
        rotated = [np.rot90(pattern, k=k) for k in range(1, 4) for pattern in patterns]
    if flip:
        patterns += flipped
    if rotate:
        patterns += rotated

    patterns = [to_tuple(pattern) for pattern in patterns]
    pattern_to_freq = {item: patterns.count(item) for item in patterns}
    return im, pattern_to_freq


def to_tuple(array):
    # todo doc
    if isinstance(array, np.ndarray):
        return tuple(map(to_tuple, array))
    else:
        return array


def to_ndarray(tup):
    # todo doc
    if isinstance(tup, tuple):
        return np.array(list(map(to_ndarray, tup)))
    else:
        return tup


# todo create a dictionary that maps each tile to its probability

def save_patterns(patters, output_path):
    """
    Saves a list of image tiles to new image files in a given output path.
    :param patters: a list of image tiles, each represented as a numpy array
    :param output_path: a string containing the path to the output directory
    """
    # Create the output directory if it doesn'tup exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate through the image tiles and save each one to a new image file
    for i, tile in enumerate(patters):
        # Convert the numpy array to a PIL image
        im = Image.fromarray(tile)

        # Save the image to the output path
        im.save(os.path.join(output_path, f'tile_{i}.jpg'))


def mask_with_offset(pattern, offset):
    """
    :param pattern: an N*N*channels ndarray
    :param offset: a 2D vector
    :return: a subarray of pattern, which is all entries
    that are inside the intersection of the pattern with another pattern offset by offset
    """
    x_offset, y_offset = offset
    if abs(x_offset) > len(pattern) or abs(y_offset) > len(pattern[0]):
        return np.array([[]])
    return pattern[max(0, x_offset):min(len(pattern) + x_offset, len(pattern)),
           max(0, y_offset):min(len(pattern[0]) + y_offset, len(pattern[0])), :]


def check_for_match(p1, p2, offset):
    """
    :param p1: first pattern
    :param p2: second pattern
    :param offset: offset of the first pattern
    :return: true if p2 equals to p1_ind with offset offset
    """
    p1_offset = mask_with_offset(p1, offset)
    p2_offset = mask_with_offset(p2, flip_dir(offset))
    return np.all(np.equal(p1_offset, p2_offset))


def get_dirs(n):
    """
    :param n: size of patterns
    :return: all coordinates around the pattern
    """
    dirs = [(i, j) for j in range(-n + 1, n) for i in range(-n + 1, n)]
    dirs.remove((0, 0))
    return dirs


def flip_dir(d):
    # todo doc
    return tuple(-1 * i for i in d)


def get_rules(patterns, directions):
    """
    creates the rules data structure, which is a list where entry i holds a dictionary that maps offset (x,y) to a set of
    indices of all patterns matching there
    :param directions: an array of all surrounding possible offsets
    :param patterns: the list of all the patterns
    :return:the rules list
    """
    rules = [{dire: set() for dire in directions} for _ in range(len(patterns))]
    for i in range(len(patterns)):
        for d in directions:
            for j in range(i, len(patterns)):
                if check_for_match(patterns[i], patterns[j], d):
                    rules[i][d].add(j)
                    rules[j][flip_dir(d)].add(i)
    return rules


def add_rule(direction, p1_ind, p2_ind, rules):
    # todo doc
    if direction in rules[p1_ind]:
        rules[p1_ind][direction].append(p2_ind)
    else:
        rules[p1_ind][direction] = {p2_ind}


def show_patterns(patterns, freq):
    # todo doc
    plt.figure(figsize=(10, 10))
    freq_sum = sum(freq)
    for m in range(len(patterns)):
        axs = plt.subplot(int(math.sqrt(len(patterns))), math.ceil(len(patterns) / int(math.sqrt(len(patterns)))), m + 1)
        axs.imshow(patterns[m])
        axs.set_xticks([])
        axs.set_yticks([])
        plt.title("weight: %.0f\nprob: %.2f" % (freq[m], freq[m] / freq_sum))
    plt.show()


def get_min_entropy_coordinates(coefficient_matrix, frequencies):
    """
    todo doc
    :param coefficient_matrix:
    :param frequencies:
    :return:
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


def is_cell_collapsed(coefficient_matrix, cell_pos):
    """
    todo doc
    :param coefficient_matrix:
    :param cell_pos:
    :return:
    """
    return np.sum(coefficient_matrix[cell_pos[0], cell_pos[1], :]) == 1


def propagate_cell(orig_cell, direction, coefficient_matrix, rules):
    """
    todo doc
    :param orig_cell:
    :param direction:
    :param coefficient_matrix:
    :param rules:
    :return:
    """
    adjacent_cell_pos = orig_cell[0] + direction[0], orig_cell[1] + direction[1]
    valid_patterns_in_adjacent_cell = coefficient_matrix[adjacent_cell_pos]
    possible_patterns_in_orig_cell = np.where(coefficient_matrix[orig_cell])[0]
    possibilities_in_dir = np.full(coefficient_matrix[orig_cell].shape, False)
    for pattern in possible_patterns_in_orig_cell:
        for possibility in rules[pattern][direction]:
            possibilities_in_dir[possibility] = True

    return np.multiply(possibilities_in_dir, valid_patterns_in_adjacent_cell)


def in_matrix(pos, dims):
    """
    todo doc
    :param pos:
    :param dims:
    :return:
    """
    return 0 <= pos[0] < dims[0] and 0 <= pos[1] < dims[1]


def propagate(min_entropy_pos, coefficient_matrix, rules, directions):
    # todo doc
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


def observe(coefficient_matrix, frequencies):
    # todo doc
    # If contradiction
    if np.any(~np.any(coefficient_matrix, axis=2)):
        return (-1, -1), coefficient_matrix, CONTRADICTION
    # Get min pos
    min_entropy_pos_x, min_entropy_pos_y, status = get_min_entropy_coordinates(coefficient_matrix, frequencies)
    # If fully collapsed
    if status == WAVE_COLLAPSED:
        return (min_entropy_pos_x, min_entropy_pos_y), coefficient_matrix, WAVE_COLLAPSED
    min_entropy_pos, coefficient_matrix = collapse_cell(coefficient_matrix, frequencies, (min_entropy_pos_x, min_entropy_pos_y))
    return min_entropy_pos, coefficient_matrix, RUNNING


def collapse_cell(coefficient_matrix, frequencies, min_entropy_pos):
    """
    todo doc
    todo rename
    :param coefficient_matrix:
    :param frequencies:
    :param min_entropy_pos:
    :return:
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
    return min_entropy_pos, coefficient_matrix


def main(input_path, pattern_size, out_width, out_height):
    # todo doc
    coefficient_matrix, directions, frequencies, patterns, rules = initialize(input_path, out_height, out_width, pattern_size)
    status = 1
    iteration = 0
    if SAVE_VIDEO:
        images = [image_from_coefficients(coefficient_matrix, patterns)[1]]
    while status != WAVE_COLLAPSED:
        min_entropy_pos, coefficient_matrix, status = observe(coefficient_matrix, frequencies)
        iteration += 1
        if status == CONTRADICTION:
            print("\nfound contradiction")
            exit(-1)
        if status == WAVE_COLLAPSED:
            print("\nwave collapsed!")
            save_iterations_to_video(images)
            return show_iteration("final", patterns, coefficient_matrix)
        if SAVE_VIDEO:
            collapsed, image = image_from_coefficients(coefficient_matrix, patterns)
            images.append(image)
            progress_bar(out_height * out_width, collapsed)
        if iteration % 20 == 0:
            show_iteration(iteration, patterns, coefficient_matrix)
        coefficient_matrix = propagate(min_entropy_pos, coefficient_matrix, rules, directions)


def initialize(input_path, out_height, out_width, pattern_size):
    directions = get_dirs(pattern_size)
    im, pattern_to_freq = get_patterns(input_path, pattern_size, flip=False, rotate=False)
    patterns, frequencies = np.array(np.array([to_ndarray(tup) for tup in pattern_to_freq.keys()])), \
                            list(pattern_to_freq.values())
    show_patterns(patterns, frequencies)
    rules = get_rules(patterns, directions)
    coefficient_matrix = np.full((out_height, out_width, len(patterns)), True, dtype=bool)
    return coefficient_matrix, directions, frequencies, patterns, rules


def show_iteration(iteration, patterns, coefficient_matrix):
    collapsed, res = image_from_coefficients(coefficient_matrix, patterns)
    plt.imshow(res)
    plt.title = f"iteration number: {iteration}, cells collapsed: {collapsed}"
    plt.show()
    return res


def save_iterations_to_video(images):
    fps = len(images)//20
    images = np.array(images).astype(np.int8)
    clip = ImageSequenceClip(list(images * 255), fps=fps)
    clip.write_gif('wfc.gif', fps=fps)


def image_from_coefficients(coefficient_matrix, patterns):
    r, c, num_patterns = coefficient_matrix.shape
    res = np.empty((r, c, 3))
    collapsed = 0
    for row in range(r):
        for col in range(c):
            valid_patterns = np.where(coefficient_matrix[row, col])[0]
            if len(valid_patterns) == 1:
                collapsed += 1
            res[row, col] = np.mean(patterns[valid_patterns], axis=0)[0, 0] / 255
    return collapsed, res


def print_adjacency_rules(rules):
    """
    todo doc
    :param rules:
    :return:
    """
    for i in range(len(rules)):
        print(f"pattern number {i}:")
        for k, v in rules[i].items():
            print(f"key: {k}, values: {v}")


def progress_bar(max, curr):
    percentage = int(100 * (curr / float(max)))
    bar = '█' * percentage + '-' * (100 - percentage)
    print(f"\r|{bar}| {round(percentage, 2)}% ", end='\b')


res = main('houses3.png', 4,80,80)

"""
 to do tests:
 1. test different pattern sizes
 2. test handling small and large inputs
 3. 
"""

# num_patterns, w, h = 9, 10, 10
# e = np.random.randint(0, 2, (w, h, num_patterns), bool)
# freq = [np.random.randint(1, 15) for _ in range(num_patterns)]
# res = observe(e, freq)
# e = np.full((w, h, num_patterns), 0, bool)
# freq = [np.random.randint(1, 15) for _ in range(num_patterns)]
# res1 = observe(e, freq)
# e = np.full((w, h, num_patterns), 0, bool)
# for i in range(w):
#     for j in range(h-1):
#         e[i, j, np.random.randint(0, num_patterns)] = True
#     e[i,9,1] = True
#     e[i,9,0] = True
# freq = [np.random.randint(1, 15) for _ in range(num_patterns)]
# res1 = observe(e, freq)
# print(1)
# e = np.full((25, 25, 9), True, dtype=bool)
# # e[10, 15, 3] = False
# e[9, 14, 3] = False
# e[1, 2, :] = False
# freq = [1, 2, 2, 4, 1, 2, 1, 2, 1]
# res = get_min_entropy_coordinates(e, freq)
