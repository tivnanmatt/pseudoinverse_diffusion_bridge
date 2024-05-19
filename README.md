
# Pseudoinverse Diffusion Bridge

## Overview
This project explores the application of pseudoinverse diffusion bridge models to various imaging modalities. The goal is to develop advanced techniques for image reconstruction and enhancement using diffusion models.

## Directory Structure
The project directory is organized as follows:

```
pseudoinverse_diffusion_bridge/
├── animations/
├── diffusion_laboratory/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── datasets.py
│   ├── diffusion_models.py
│   ├── measurement_models.py
│   ├── networks.py
│   ├── sample.py
│   ├── train.py
│   └── utils.py
├── figures/
├── weights/
└── README.md
```

## Components

### 1. Animations
This directory contains generated animations showcasing the performance of various diffusion models on different datasets.

### 2. Diffusion Laboratory
This is the core directory containing all the code for the project. Below are the details of each file:

- `__init__.py`: Initializes the diffusion laboratory module.
- `cli.py`: Command-line interface for training and sampling with diffusion models.
- `config.py`: Configuration setup for the project including dataset paths, model parameters, etc.
- `datasets.py`: Code to load and preprocess different datasets.
- `diffusion_models.py`: Implementation of various diffusion models including pseudoinverse diffusion bridge.
- `measurement_models.py`: Models to simulate measurement processes and noise.
- `networks.py`: Neural network architectures used in the project.
- `sample.py`: Functions for sampling from trained models.
- `train.py`: Training routines for the diffusion models.
- `utils.py`: Utility functions for the project.

### 3. Figures
Contains visual results of the experiments, such as plots of reverse diffusion processes and reconstructed images.

### 4. Weights
Directory for storing the trained model weights for different diffusion models and datasets.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Matplotlib
- NumPy

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/pseudoinverse_diffusion_bridge.git
    cd pseudoinverse_diffusion_bridge
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Code

#### Training a Model
To train a model, use the `cli.py` script with the `--train` flag:
```sh
python diffusion_laboratory/cli.py --train --config path/to/config.yaml
```

#### Sampling from a Model
To sample from a trained model, use the `cli.py` script with the `--sample` flag:
```sh
python diffusion_laboratory/cli.py --sample --config path/to/config.yaml
```

### Configuration
The `config.yaml` file contains all the necessary configuration for training and sampling, such as paths to datasets, model parameters, and training settings.

## Contributing
Contributions are welcome! Please read the [contributing guide](CONTRIBUTING.md) to get started.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
We would like to thank all contributors and the open-source community for their valuable work.

