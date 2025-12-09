# [Term Project] AlexNet-based Histopathological Image Classification

## 1. Introduction

The goal of this project is to implement and evaluate various deep learning models based on the AlexNet architecture for classifying histopathological tissue images. Through a series of systematic experiments, we explore different model architectures, data preprocessing techniques, and feature extraction methods to find the optimal combination for this specific task.

This repository contains the complete source code, experimental results, and documentation for the project.

---

## 2. Dataset

- **Dataset:** Proof-of-Concept (POC) Histopathological Image Dataset
- **Task:** Multi-class image classification
- **Classes (4 types):**
  1.  `Chorionic_villi`
  2.  `Decidual_tissue`
  3.  `Hemorrhage`
  4.  `Trophoblastic_tissue`
- **Structure:**
  - The data is split into `train` and `test` sets.
  - Each set contains subdirectories for the four classes, with corresponding image files.
- **Initial Analysis (`check.ipynb`):**
  - Images in the `train` directory have varying sizes.
  - All images in the `test` directory are uniformly sized at `(224, 224)`.
  - PCA (Principal Component Analysis) was performed on the training images' RGB values to calculate eigenvalues and eigenvectors for "Fancy PCA" color augmentation, a technique mentioned in the original AlexNet paper.

---

## 3. Project Structure

The repository is organized as follows:

```
POC_AlexNet/
├── configs/              # YAML configuration files for each experiment
├── data/                 # Dataset directory
│   └── poc_dataset/
│       ├── train/
│       └── test/
├── saved_models/         # Saved model weights (.pth) and predictions (.pt)
├── src/                  # Source code
│   ├── data_loader.py    # Data loading and preprocessing pipelines
│   ├── engine.py         # Core training and evaluation loops
│   └── models.py         # Model architecture definitions (AlexNet variants, CBAM)
├── experiment_1.ipynb    # Notebook for Model Architecture experiments
├── experiment_2.ipynb    # Notebook for Preprocessing experiments
├── experiment_3.ipynb    # Notebook for Feature Extraction & ML Ensemble
├── train.py              # Main script to run training from the command line
└── README.md             # Project documentation (this file)
```

---

## 4. Methodology & Experiments

We conducted a series of three main experiments to systematically improve and evaluate the model's performance.

### Experiment 1: Model Architecture Exploration

- **Objective:** To identify the most effective AlexNet-based architecture for the given task.
- **Models Compared:**
  - `exp1a`: Baseline AlexNet trained from scratch.
  - `exp1b`: AlexNet with pre-trained ImageNet weights.
  - `exp1c`: Pre-trained AlexNet with **Batch Normalization (BN)** added after each convolutional/fully-connected layer.
  - `exp1d`: Pre-trained AlexNet with a **CBAM (Convolutional Block Attention Module)** added before each pooling layer.
  - `exp1e`: Pre-trained AlexNet combining both **BN and CBAM**.
- **Conclusion:** Although the standard pre-trained model (`exp1b`) showed the highest peak accuracy, its training and validation loss were highly volatile. The `exp1e` model (`AlexNet + BN + CBAM`) was selected as the optimal architecture, as it achieved the second-highest accuracy while demonstrating significantly more stable learning.

### Experiment 2: Preprocessing & Data Augmentation

- **Objective:** To compare the impact of different image preprocessing methods on the performance of the best models from Experiment 1.
- **Methods Compared:**
  - **`Resize`:** Simply resizing the input image to 224x224.
  - **`10-Crop`:** As proposed in the AlexNet paper, images are resized to 256x256. During training, random 224x224 crops are used for data augmentation. During testing, predictions from 10 crops (center, four corners, and their horizontal flips) are averaged.
- **Conclusion:** The `10-Crop` method consistently outperformed the simple `Resize` method. This is attributed to its dual benefit of providing effective data augmentation during training and enabling more robust predictions through ensemble-like averaging at test time.

### Experiment 3: Feature Extraction & ML Ensemble

- **Objective:** Inspired by the paper *"Hierarchical Deep Feature Fusion and Ensemble Learning"*, this experiment explores using the best deep learning model as a feature extractor for traditional machine learning classifiers.
- **Methodology:**
  1.  **Feature Extraction:** Used the best model from the previous experiments (`AlexNet + BN + CBAM` with `10-Crop`) to extract features from two different points:
      - The last fully connected layer (`extract_fc7`).
      - A fusion of features from the last convolutional layer and the last fully connected layer (`extract_fused`).
  2.  **ML Classifiers:** The extracted features were used to train three different ML models:
      - MLP (Multi-layer Perceptron)
      - KNN (K-Nearest Neighbors)
      - SVM (Support Vector Machine with RBF kernel)
  3.  **Ensemble:** The final prediction was generated by a weighted average of the probabilities from the three ML models, with weights determined by their 5-fold cross-validation scores.
- **Conclusion:** This approach demonstrated that features learned by the deep model could be effectively used by simpler classifiers. The ensemble of ML models on fused features (`extract_fused`) achieved the highest accuracy of **86.23%**.

---

## 5. Results Summary

| Experiment          | Model Architecture     | Preprocessing | Best Accuracy |
| ------------------- | ---------------------- | ------------- | ------------- |
| 1a (Baseline)       | AlexNet                | Resize        | 78.69%        |
| 1b (Pre-trained)    | AlexNet                | Resize        | **87.29%**    |
| 1c (+BN)            | AlexNet + BN           | Resize        | 85.11%        |
| 1d (+CBAM)          | AlexNet + CBAM         | Resize        | 83.79%        |
| 1e (+BN+CBAM)       | AlexNet + BN + CBAM    | Resize        | 85.90%        |
| 2c (from 1d)        | AlexNet + CBAM         | **10-Crop**   | 83.12%        |
| 2d (from 1e)        | AlexNet + BN + CBAM    | **10-Crop**   | 85.44%        |
| 3 (Ensemble)        | Feature Fusion         | 10-Crop       | **86.23%**    |

The most stable and robust single model was **`AlexNet + BN + CBAM` with `10-Crop` preprocessing**. The highest overall performance was achieved in Experiment 3 by using the features from this model to train an **ensemble of ML classifiers**.

---

## 6. How to Run

### a. Dependencies

It is recommended to set up a virtual environment. The required libraries can be installed via `pip`. While a `requirements.txt` file is not provided, the key dependencies are:

- `torch`
- `torchvision`
- `pyyaml`
- `scikit-learn`
- `matplotlib`
- `numpy`
- `tqdm`

### b. Running Experiments

You can replicate the experiments by running the `train.py` script with the appropriate configuration file. The configuration files for all experiments are located in the `configs/` directory.

**Example:** To run the baseline experiment (`exp1a`):
```bash
python train.py --config configs/exp1_a.yaml
```

To run the final model from Experiment 2 (`exp2d`):
```bash
python train.py --config configs/exp2_d.yaml
```

### c. Viewing Results

The Jupyter Notebooks (`experiment_1.ipynb`, `experiment_2.ipynb`, `experiment_3.ipynb`) contain the code to run each experiment and visualize the results. You can run these notebooks to see the detailed process and output for each experimental stage.
