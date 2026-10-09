# Machine Learning Lab 6: Artificial Neural Network (ANN) Classification

## Overview

This project demonstrates a multiclass classification task using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**. The model uses the Seeds dataset to classify wheat seeds into one of three classes based on their measured features.

The notebook includes data inspection, exploratory data analysis (EDA), feature scaling, ANN model construction and training, and evaluation on a held-out test set.

## Aim

To build and evaluate an Artificial Neural Network model for classifying wheat seeds into three classes using their numerical features.

## Dataset

- **Dataset:** Seeds dataset (`seeds.csv`)
- **Target column:** `Class (1, 2, 3)`
- **Task:** Multiclass classification
- **Input features:** Numerical measurements of wheat seeds, including features such as area and perimeter.

Place `seeds.csv` in the same directory as the Jupyter Notebook before running it. The notebook expects the target column to be named exactly `Class (1, 2, 3)`.

## Technologies and Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

## Workflow

1. **Import libraries** — load the Python libraries required for data handling, visualization, preprocessing, and neural-network modelling.
2. **Load and inspect data** — read `seeds.csv` and examine sample rows, dataset information, descriptive statistics, dimensions, missing values, and duplicate records.
3. **Exploratory Data Analysis** — visualize class distribution, feature histograms, boxplots, a correlation heatmap, an Area-versus-Perimeter scatter plot, and pairwise feature relationships.
4. **Prepare features and labels** — separate the input features (`X`) from the target class (`y`). Convert class labels from `1, 2, 3` to `0, 1, 2` for the model.
5. **Split the dataset** — create training and testing subsets using an 80:20 split.
6. **Scale features** — use `StandardScaler`, fitting it on the training data and applying the same transformation to the test data.
7. **Build the ANN** — create a Sequential neural network with an input layer for seven features, two hidden Dense layers using ReLU activation, and a three-neuron Softmax output layer.
8. **Compile and train** — use the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric. Train for up to 150 epochs with a batch size of 16, using a validation split and early stopping.
9. **Evaluate the model** — evaluate test loss and accuracy and plot training/validation accuracy and loss.
10. **Generate predictions** — obtain class probabilities, select the predicted class with `argmax`, convert predicted labels back to the original class numbering, and compare actual versus predicted classes.

## ANN Architecture

| Layer | Configuration |
|---|---|
| Input | 7 features |
| Hidden layer 1 | 16 neurons, ReLU activation |
| Hidden layer 2 | 8 neurons, ReLU activation |
| Output layer | 3 neurons, Softmax activation |

**Training configuration**

- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: `sparse_categorical_crossentropy`
- Batch size: `16`
- Maximum epochs: `150`
- Validation split: `20%`
- Early stopping: monitors validation loss, patience of `10`, and restores the best weights

## Evaluation and Visualizations

The notebook produces:

- Class-distribution pie chart
- Feature histograms
- Boxplots for feature distributions and potential outliers
- Feature-correlation heatmap
- Area-versus-Perimeter scatter plot
- Pairplot grouped by seed class
- Training and validation accuracy curves
- Training and validation loss curves
- Test loss and test accuracy
- A table comparing actual and predicted classes

The final performance depends on the dataset and training run. Use the values printed by your own notebook for any reported accuracy; no fixed accuracy is assumed here.

## Installation

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## How to Run

1. Clone or download this repository.
2. Ensure `ML_Lab_6.ipynb` and `seeds.csv` are in the same folder.
3. Install the dependencies listed above.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `ML_Lab_6.ipynb`.
6. Run the notebook cells from top to bottom.

> **Note:** The notebook loads the dataset with `pd.read_csv("seeds.csv")`. If the CSV is stored elsewhere, update the path accordingly.

## Expected Outcome

The notebook trains an ANN to predict one of three wheat-seed classes. It reports the test loss and accuracy and displays a comparison of actual and predicted class labels. Run the notebook to obtain the results for your environment.

## Repository Structure

```text
.
├── ML_Lab_6.ipynb
├── seeds.csv
└── README.md
```

## Learning Outcomes

After completing this lab, you should be able to:

- Perform basic data inspection and exploratory data analysis.
- Prepare features and labels for a multiclass classification problem.
- Apply standardization using `StandardScaler`.
- Design and compile a feed-forward ANN using TensorFlow/Keras.
- Train a model with validation data and early stopping.
- Evaluate classification performance and interpret training curves.
- Generate predictions and compare them with actual labels.

## Author

**Shreya Sharma**

## License

This project is intended for educational and academic use.
