import ntpath
from typing import List, Tuple

import numpy as np
from PIL import Image as Img
from matplotlib import pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

DEFAULT_FPS = 30  # Fps of the output video
DEFAULT_VIDEO_LENGTH = 6  # Length of the output video
DEFAULT_OUT_VID_HEIGHT = 1000  # Vertical size (in pixels) of the output video, preserves the original aspect ratio
NUM_OF_ITERATIONS_BETWEEN_RENDER = 15  # The number of iterations between rendering in runtime


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
