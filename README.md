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
given a small image as its input, such that at any small enviroment in the
output image, there is a resemblance to the input image.

This implementation of the algorithm is written with functional programming
design in mind, and enables runtime rendering of the collapse process,
pattern extraction and viewing (unrelated to the actual algorithm run), and
saving the result of each algorithm run both as an image of the final
product and as a video of the collapse process itself.

## Table of context

- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#Usage)
- [Features](#features)
- [Adding new Music Videos](#adding-new-music-videos)

## Demo

<div align="center">
<img src="https://github.com/avihuxp/WaveFunctionCollapse/blob/master/README/wfc.gif?raw=true" alt="random_forest_error" width="800"/>
</div>

For a full demo video check this [link](https://youtu.be/Ten6MIWd2DA).

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Requirements

jukebox requires the followin to run:

- Python 3.7.3+

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

### Installation

1. Clone the repo
   ```bash
   git clone https://github.com/avihuxp/WaveFunctionCollapse.git
   ```
2. Install the required packages - using the supplied requirements.txt, run:
      ```bash
      pip install -r requirements.txt
      ```

### Usage

Run the program, with the following usage:

```bash
 python3 wave_function_collapse.py <path_to_input_image> <pattern_size> <out_height> <out_width> [<flip> <rotate>]
```

where parameters are:

1. **`path_to_input_image`** - `str`: The path to the imput image for the
   algoritm, i.e. the base pattern.
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

See
the [open issues](https://github.com/avihuxp/WaveFunctionCollapse/issues)
for a full list of proposed features (and known issues).

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Copyright

MIT License

Copyright (c) 2022 avihuxp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.