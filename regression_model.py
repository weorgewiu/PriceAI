import torch
import numpy as np
import pandas as pd
import kaggle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Import Data
df = pd.read_csv("kaggle/train.csv")
df_test = pd.read_csv("kaggle/test.csv")

x = df.iloc[:, :-1]
y = df.iloc[:, -1] 

x_test = df_test.copy()

# Fill missing values like training set
for col in x_test.select_dtypes(include=["float64", "int64"]).columns:
    if x_test[col].isnull().any():
        median_val = x_test[col].median()
        x_test[col] = x_test[col].fillna(median_val)

for col in x_test.select_dtypes(include=["object"]).columns:
    if x_test[col].isnull().any():
        x_test[col] = x_test[col].fillna("None")

#Preprocessing: Fill in NA
#If Numerical, fill with median value
for col in x.select_dtypes(include=["float64", "int64"]).columns:
    if x[col].isnull().any():
        median_val = x[col].median()
        x[col] = x[col].fillna(median_val)

#If Categorical, fill with None
for col in x.select_dtypes(include=["object"]).columns:
    if x[col].isnull().any():
        x[col] = x[col].fillna("None")

cat_cols = x.select_dtypes(include=['object']).columns

x = pd.get_dummies(x, columns=cat_cols, dtype=int)
x_test = pd.get_dummies(x_test, columns= x_test.select_dtypes(include=["object"]).columns, dtype=int)

# Align test set to training set
x_test = x_test.reindex(columns=x.columns, fill_value=0)

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_dim4, hidden_dim5)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc6 = nn.Linear(hidden_dim5, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = self.fc6(x)
        return x
    
def evaluate(model, x_val, y_val):
    model.eval()
    with torch.no_grad():
        preds = model(x_val)
        
        mse = F.mse_loss(preds, y_val).item()
        rmse = mse ** 0.5
    return mse, rmse

#print (x.head())    
#print(y.head()) 

#Standard Transform
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_test_scaled = scaler.transform(x_test)

#Log Transform Y
y_log = np.log1p(y)

# Split into train and validation (80-20 split)
x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(
    x_scaled, y_log.values.reshape(-1,1), test_size=0.2, random_state=42
)

# Convert to Torch tensors
x_train = torch.tensor(x_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

x_val = torch.tensor(x_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)

model = RegressionModel(input_dim=304, hidden_dim1= 512, hidden_dim2= 256, hidden_dim3=128, hidden_dim4=64, hidden_dim5= 32, output_dim=1, dropout_rate=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

criterion = nn.MSELoss()

epochs = 20001

for epoch in range(epochs):
    model.train()

    preds = model(x_train)
    loss = criterion(preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")



rmse_log = torch.sqrt(loss)
print(f"Log RMSE: {rmse_log.item():.4f}")

mse_val, rmse_val = evaluate(model, x_val, y_val)
print(f"Validation MSE: {mse_val:.4f}")
print(f"Validation RMSE: {rmse_val:.4f}")

model.eval()
with torch.no_grad():
    test_preds_log = model(x_test_tensor)
    test_preds = torch.expm1(test_preds_log).numpy() 

submission = pd.DataFrame({
    'Id': df_test['Id'],
    'SalePrice': test_preds.flatten()
})
submission.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'")
