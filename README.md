<!-- PROJECT LOGO (light) -->
![GitHub-Mark-Light](https://github.com/avihuxp/WaveFunctionCollapse/blob/master/README/Wave%20function%20collapse.png)

<div align="center">
<div align="center">

  <p align="center">
    A Generative Model For Random Image And Pattern Extrapolation</p>
</div>

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/avihuxp/WaveFunctionCollapse?color=orange)
![GitHub issues](https://img.shields.io/github/issues/avihuxp/WaveFunctionCollapse?color=yellow)
![GitHub pull requests](https://img.shields.io/github/issues-pr/avihuxp/WaveFunctionCollapse?color=yellow)
![GitHub repo size](https://img.shields.io/github/repo-size/avihuxp/WaveFunctionCollapse)
![GitHub](https://img.shields.io/github/license/avihuxp/WaveFunctionCollapse)

<a href="https://github.com/avihuxp/WaveFunctionCollapse/issues">Report Bug</a>
·
<a href="https://github.com/avihuxp/WaveFunctionCollapse/issues">Request
Feature</a>
</div>

## About the project

This project is my implementation of the "Wave Function Collapse" algorithm.
The program is designed to generate an output pattern of arbitrary size,
given a small image as its input, such that at any small environment in the
output image, there is a resemblance to the input image.

This implementation of the algorithm is written with functional programming
design in mind, and enables runtime rendering of the collapse process,
pattern extraction and viewing (unrelated to the actual algorithm run), and
saving the result of each algorithm run both as an image of the final
product and as a video of the collapse process itself.

## Table of context

- [Demo](#demo)
- [Wave Function Collapse explained](#Wave-Function-Collapse-explained)
- [Requirements](#Requirements)
- [Installation](#installation)
- [Usage](#Usage)
- [Features](#features)
- [Resources](#Resources-and-Acknowledgments)

## Demo

<div align="center">
<img src="https://github.com/avihuxp/WaveFunctionCollapse/blob/master/README/wfc.gif?raw=true" alt="random_forest_error" width="800"/>
</div>

For a full demo video check this [link](https://youtu.be/Ten6MIWd2DA).

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Wave Function Collapse explained 

The "wave function collapse" algorithm, as it is used in computer science
and game design, is a technique that generates random but coherent
sequences of events or states (in our case, this is the output image). The
algorithm is based on the idea of the wave function collapse from quantum
mechanics, where a system in a superposition of states collapses into a single
definite state.

The input parameters for the algorithm include the initial state of the
system (in our case, all patterns are valid in all cells), a set of rules for
the possible transitions between states, and a probability distribution for
those transitions (as gathered from the input data). The algorithm proceeds in
steps, in each step a new state is selected according to the transition rules
and probability distribution.

The stages of the algorithm are as follows:

1. **Initialization:** This step can be broken into 3 parts:
    - The algorithm interprets the input, and gathers from it all possible
      states and patters for each cell of the wave.
    - The algorithm gathers the transition rules, meaning which patterns or
      states can relate to each other and in what way.
    - The 'wave' (which is a matrix with width and height of the output, and
      with depth corresponding to the number of patterns gathered) is
      initiated, with all patterns valid in all the cells.
2. **Observe:** A new state (a specific pattern in a specific cell) with
   the minimal entropy (i.e. the minimal number of patterns possible) is
   selected according to the transition rules and probability distribution:
   In case of a draw in entropy, the algorithm will choose a cell randomly
   from all cells with minimal entropy. After a cell with minimal entropy
   is found, it is collapsed (a pattern is chosen for it, and it is no
   longer in superposition) using the probability distribution of the
   patterns.
3. **Propagate:** The wave's state is now updated, with all cells affected
   by the collapse of the cell in the previous step updated to hold only
   valid patterns.
4. **Repeat:** The algorithm continues to repeat steps 2 and 3 until all
   cells - and by that the wave - are fully collapsed.
   The output of the algorithm is the collapsed wave, meaning that at each
   cell, only one pattern is valid, and will be selected for the output image.

This algorithm is widely used in game design, to generate random but coherent
levels, items, enemies, etc. in a game, but also in other fields such as
natural language processing, music generation, art, and many others.
For further reading, I
recommend [this blogpost](https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/)
by Robert Heaton.

## Requirements

The program requires the following to run:

- Python 3.7.3+

## Installation

1. Clone the repo
   ```bash
   git clone https://github.com/avihuxp/WaveFunctionCollapse.git
   ```
2. Install the required packages - using the supplied requirements.txt, run:
      ```bash
      pip install -r requirements.txt
      ```

## Usage

Run the program, with the following usage:

```bash
 python3 wave_function_collapse.py <path_to_input_image> <pattern_size> <out_height> <out_width> [<flip> <rotate>]
```

where parameters are:

1. **`path_to_input_image`** - `str`: The path to the input image for the
   algorithm, i.e. the base pattern.
2. **`pattern_size`** - `int`: The size of the square sub images of the input
   image, should be as small as possible for efficiency, and large
   enough to capture the largest basic feature in the input image.
3. **`out_height \ out_width`** - `int`s: the size, in pixels, of the
   output of the program.

***Optional parameters***

1. **`flip`** -  `bool`: Default is `False`, if `True`, the output
   will be able to include flipped (horizontally and vertically)
   versions of every pattern extracted from the input image.
2. **`rotate`** -  `bool`: Default is `False`, if `True`, the output
   will be able to include rotated (by 90&deg;, 180&deg;, and 270&deg;)
   versions of every pattern extracted from the input image.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Features

- [x] Generate outputs from both RGB and GrayScale images.
- [x] Render images of the collapse progress in runtime using matplotlib.
- [x] Output the result of the algorithm to a large size image for better
  viewing experience.
- [ ] Improved runtime performance
- [ ] Added support for initial wave states (i.e. letting the wave function
  know about sky / ground / walls in the input image).
- [ ] Wrap around support for pattern extraction.
- [ ] Add a "Stride" option for pattern extraction, rules initialization,
  and propagation, so that this code will be able to run the tiled variant
  of the WFC algorithm.

See
the [open issues](https://github.com/avihuxp/WaveFunctionCollapse/issues)
for a full list of proposed features (and known issues).

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Resources and Acknowledgments

This project is heavily inspired and implemented with the help of the
following sources:

1. The original WaveFunctionCollapse implementation, which can be found in
   [Maxim Gumin's repo](https://github.com/mxgmn/WaveFunctionCollapse).
2. [The paper](https://adamsmith.as/papers/wfc_is_constraint_solving_in_the_wild.pdf)
   based on mxgmn's work.
3. The Coding Train's video
   on [WaveFunctionCollapse](https://www.youtube.com/watch?v=rI_y2GAlQFM&t=1387s)
   , which introduced me to the algorithm,
   though it implements the tiled variant of the algorithm.

Also, [OrrMatzkin](https://github.com/OrrMatzkin) deserves a noteable
mention for the help with the README file.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Copyright

MIT License

Copyright (c) 2023 avihuxp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>