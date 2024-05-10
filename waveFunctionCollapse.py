import sys
from src.main import wave_function_collapse

USAGE_MSG = "Error: invalid input_examples. Usage: python3 wave_function_collapse.py <input_path> <pattern_size> " \
            "<out_height> <out_width> [flip] [rotate] [render_iterations] [render_video]"

if __name__ == "__main__":
    try:
        if len(sys.argv) > 9:
            raise IndexError
        input_path, pattern_size, out_height, out_width = sys.argv[1:5]
        pattern_size, out_height, out_width = int(pattern_size), int(out_height), int(out_width)
        flip = sys.argv[5] == 'True' if len(sys.argv) >= 6 else False
        rotate = sys.argv[6] == 'True' if len(sys.argv) >= 7 else False
        render_iterations = sys.argv[7] == 'True' if len(sys.argv) >= 8 else True
        render_video = sys.argv[8] == 'True' if len(sys.argv) == 9 else True
    except (TypeError, ValueError, IndexError):
        print(USAGE_MSG)
    else:
        wave_function_collapse(input_path, pattern_size, out_height, out_width, flip, rotate, render_iterations,
                               render_video)
