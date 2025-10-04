# House Prices: Advanced Regression Techniques (PyTorch)

This repository contains a PyTorch-based solution for the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
The model predicts home sale prices using a deep neural network with preprocessing, feature engineering, and log-transformed targets.

---

## Features

* Handles missing values with median imputation (numerical) and `"None"` (categorical).
* One-hot encodes categorical features and aligns test features to training.
* Scales input features with `StandardScaler`.
* Applies log-transform to the target (`SalePrice`) to reduce skewness.
* Deep feedforward regression model with multiple hidden layers and dropout.
* Optimized with Adam and adaptive learning rate scheduling (`ReduceLROnPlateau`).
* Generates a Kaggle-compatible `submission.csv` file.

---

## Requirements

Install the dependencies with:

```bash
pip install torch pandas numpy scikit-learn matplotlib kaggle
```

---

## Data

Download the dataset from Kaggle:

Included in the repository are `train.csv` and `test.csv`.

---

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/weorgewiu/PriceAI.git
   cd PriceAI
   ```

2. Run the training script:

   ```bash
   python regression_model.py
   ```

3. After training, the model will output:

   * Training and validation loss logs.
   * Final `submission.csv` containing predictions.

---

## Model Architecture

* Input layer: matches number of processed features (~300).
* Hidden layers: 512 → 256 → 128 → 64 → 32 units (ReLU activations, dropout).
* Output layer: 1 neuron for regression.
* Loss: Mean Squared Error (MSE).
* Evaluation metric: Root Mean Squared Error (RMSE).

---

## Results

* Achieves reasonable validation RMSE consistent with strong Kaggle baseline models.
* Predictions are log-transformed back to actual prices for submission.

---

## Submission

The script generates a file:

```
submission.csv
```

in the format required by Kaggle:

```
Id,SalePrice
1,123456.78
2,98765.43
...
```

You can submit this file directly to the competition leaderboard.

---

## License

This project is licensed under the MIT License.
