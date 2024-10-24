# LeafAreaCalc

![LeafAreaCalc Logo](images/logo.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

**LeafAreaCalc** is a comprehensive tool designed for accurate leaf area classification from digital images. Whether you're a botanist, researcher, or hobbyist, LeafAreaCalc provides an intuitive interface and powerful processing capabilities to streamline your leaf analysis workflow.

![Sample Leaf Image](images/sample_leaf.jpg)

## Features

- **Adaptive Thresholding in RGB Space:** Enhance leaf detection by excluding near-white non-leaf areas immediately after cropping.
- **Configurable Cropping:** Adjust cropping percentages to focus on specific regions of interest within your images.
- **Contrast Adjustment:** Optionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve image contrast.
- **Morphological Operations:** Refine binary images using morphological closing with adjustable kernel sizes and iterations.
- **Contour Detection:** Accurately identify and count leaf contours, calculating their respective areas.
- **Mean RGB Calculation:** Compute average RGB values within detected leaf areas for color analysis.
- **Graphical User Interface (GUI):** User-friendly interface for configuring settings, selecting directories, and monitoring processing.
- **Logging:** Comprehensive logging with adjustable verbosity levels to track processing steps and debug issues.
- **Image Preview:** Preview images directly within the GUI for easy verification.
- **Configuration Management:** Save and reset configurations with ease, ensuring reproducible results.

![GUI Screenshot](images/gui_screenshot.png)

## Installation

### Prerequisites

- **Python 3.6+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **Virtual Environment (Recommended)**: It's advisable to use a virtual environment to manage dependencies.

### Steps

1. **Clone the Repository**
   
   ```bash
   git clone https://github.com/yourusername/LeafAreaCalc.git
   cd LeafAreaCalc
   ```


2. **Create and Activate Virtual Environment**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On Unix or MacOS
    source venv/bin/activate
    ```

