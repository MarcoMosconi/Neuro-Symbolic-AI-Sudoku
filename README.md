# Neuro-Symbolic AI for Sudoku Validation

A computer vision and neuro-symbolic AI project for classifying Sudoku grids as valid or invalid. The project explores conventional CNN-based classification and a neuro-symbolic approach that combines visual digit recognition with symbolic reasoning.

## Overview

The project investigates two approaches to Sudoku grid validation:

- **CNN-based classification:** A convolutional neural network predicts whether an entire Sudoku grid is valid or invalid from its image.
- **Neuro-symbolic classification:** A CNN recognizes digits in individual cells, while Scallop applies logical rules to detect conflicts across rows, columns, and 3×3 blocks.

The goal is to explore how combining neural perception with symbolic reasoning can address a structured visual classification task.

## Models

### 1. Conventional CNN

Implemented in `CNN_simple.ipynb`, this model directly classifies an entire Sudoku grid image into one of two categories: `valid` or `invalid`.

### 2. Neuro-symbolic CNN + Scallop

Implemented in `CNN_scallop (1).ipynb`, this approach uses a CNN to predict cell-level digits and Scallop to reason about Sudoku constraints.

The model checks for:
- Duplicate digits within a row.
- Duplicate digits within a column.
- Duplicate digits within a 3×3 block.

The symbolic rules determine whether the recognized grid contains any conflicts.

### 3. Combined model

Implemented in `CNN_combined (1).ipynb`, this model uses a CNN to recognize digits in all 81 cells of a Sudoku grid and applies Scallop rules to produce a grid-level classification.

The cell classifier contains approximately 1.18 million trainable parameters. The model was trained for 200 epochs using the Adam optimizer and binary cross-entropy loss.

## Dataset

The dataset contains **780 grayscale Sudoku grid images**, each with a resolution of 400 × 400 pixels.

The dataset is organized into two classes:

```text
sudoku_grids_dataset/
├── invalid/
└── valid/
```

Images are loaded using `torchvision.datasets.ImageFolder`, converted to grayscale tensors, and normalized before training.

The notebooks use an 80/20 random train-test split.

## Repository Structure

```text
.
├── create_dataset.py
├── grid_extraction/
│   ├── blur_detection.py
│   ├── detectorClass.py
│   ├── hough_line_detection.py
│   └── lapliacian_detection.py
├── image_generation/
│   ├── backgroundGenerator.py
│   ├── gridTransformer.py
│   └── sudokuCorruptor.py
├── models/
│   ├── CNN_combined (1).ipynb
│   ├── CNN_scallop (1).ipynb
│   ├── CNN_simple.ipynb
│   ├── combined_classifier.pt
│   ├── scallop_classifier.pt
│   └── simple_classifier.pt
└── sudoku_grids_dataset/
    ├── invalid/
    └── valid/
```

## Technologies

- Python
- PyTorch
- Torchvision
- Scallop
- NumPy
- Matplotlib
- scikit-learn
- Seaborn

## Getting Started

### Requirements

Install the required Python packages in an environment compatible with your setup:

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn tqdm
```

The neuro-symbolic notebooks additionally require Scallop. The original notebook uses a Scallop wheel built for Python 3.11 on Linux:

```bash
pip install https://github.com/pmosconi/scallop/raw/master/target/wheels/scallopy-0.2.5-cp311-cp311-manylinux_2_34_x86_64.whl
```

This wheel is platform- and Python-version-specific.

### Running the notebooks

1. Clone or download the repository.
2. Ensure the `sudoku_grids_dataset/` directory is available.
3. Open the desired notebook in Jupyter or Google Colab.
4. Set `dataset_path` to the dataset directory.
5. Run the notebook cells in order.
