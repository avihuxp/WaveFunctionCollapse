import numpy as np

from src.initiatilzation import initialize
from src.iteration import WAVE_COLLAPSED, observe, CONTRADICTION, propagate
from src.rendering import image_from_coefficients, progress_bar, show_iteration, save_iterations_to_video, \
    save_collapsed_wave, NUM_OF_ITERATIONS_BETWEEN_RENDER

WAVE_COLLAPSED_MSG = "\nwave collapsed!"
FOUND_CONTRADICTION_MSG = "\nfound contradiction"


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
