# Multi-label Image Classification for Space Radiation Effects on Cells

## Introduction
**Supervise Me** is a Machine Learning project aimed at classifying images of mice cell damages caused by space radiation. Utilizing the BPS Mice Microscopy Dataset, our models predict the type and dosage of radiation that caused the observed DNA damage. This research contributes to understanding the health hazards astronauts face, including DNA damage and immune system effects.

## Technologies Used
- **Python**: For scripting and model development
- **PyTorch/PyTorch Lightning**: For building and training neural network models
- **Weights and Biases (WandB)**: For hyperparameter tuning and performance tracking
- **AWS boto3 API**: For dataset retrieval

## Features
- **Image Classification**: Using Convolutional Neural Networks (CNNs) to classify radiation types and dosages from cell images.
- **Data Preprocessing**: Image normalization, augmentation, and tensor transformation for model training.
- **Multi-label Classification**: Extending the model to predict both radiation type and dosage.

## Installation
1. Clone the repository: `git clone https://github.com/UC-Irvine-CS175/final-project-supervise-me`
2. Install dependencies: Navigate to the environment setup directory and run `pip install -r requirements.txt`

## Usage
- Run the data retrieval script: `python src/dataset/bps_datamodule.py`
- For training the model: `python src/models/lenet_scratch.py` or `python src/models/mlp_model.py`
- Hyperparameters can be adjusted within the script as needed.

## Contributing
Contributions to enhance the model's accuracy or extend its capabilities are welcome.

## Acknowledgements
Special thanks to NASA GeneLab, Dr. Lauren Sanders, and our team mentor Nadia Ahmed for their invaluable contributions to this project.
