# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# %%
df = pd.read_csv("insurance_claims.csv")
# %%
# first 5 rows of the dataset
df.head()
# %%
# last five dataset
df.tail()
# %%
# replacing value ? with nan
df.replace('?', np.nan, inplace=True)
# %%
df.describe()
# %%
df.info()
# %%
df.isna().sum()
# %%
# Drop columns not useful for prediction
df = df.drop(['policy_number', 'policy_bind_date', 'incident_date', 'incident_location'], axis=1)

# Convert target to binary
df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

# Replace '?' with NaN and drop or fill
df.replace('?', np.nan, inplace=True)

# Drop rows with any missing values for simplicity
df.dropna(inplace=True)

# Separate features and target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']
# %%
from sklearn.preprocessing import OneHotEncoder
import joblib


# Define categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Initialize and fit encoder
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X[categorical_cols])

# After encoding training data
encoded_columns = X_cat_encoded.columns.tolist()
joblib.dump(encoded_columns, 'encoded_columns.pkl')


# Save the encoder
joblib.dump(encoder, 'encoder.pkl')

# Combine with numeric data
X_numeric = X.drop(columns=categorical_cols)
X_encoded = np.hstack([X_numeric.values, X_cat_encoded])
# Save original feature names before scaling and PCA
original_feature_names = list(X_numeric.columns) + list(encoder.get_feature_names_out(categorical_cols))
joblib.dump(original_feature_names, 'original_feature_names.pkl')

joblib.dump(categorical_cols, 'categorical_cols.pkl')
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
joblib.dump(scaler, 'scaler.pkl')
print("Feature shape after encoding and scaling:", X_scaled.shape)
# %%
# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("After SMOTE:")
print("X shape:", X_resampled.shape)
print("Class distribution:\n", y_resampled.value_counts())

# %%

# Apply PCA to keep 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_resampled)
joblib.dump(pca, 'pca.pkl')
print("Original shape:", X_resampled.shape)
print("Reduced shape:", X_pca.shape)

# %%
print(X_pca.shape)
# %%
# outlier is there so we need to Sandarise those columns
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.25, random_state=42)
# %% raw
# modelling
# %%
# SVM Model
# %%
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svc_model_train_acc = accuracy_score(y_train, svc_model.predict(X_train))
svc_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy:", svc_model_train_acc)
print("Testing Accuracy:", svc_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %%
#  decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Decision Tree model with slight tuning
dt_model = DecisionTreeClassifier(
    max_depth=10,  # Prevent overfitting
    min_samples_split=5,  # Prevent overfitting
    min_samples_leaf=2,  # Ensure generalization
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate model
decision_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
decision_test_acc = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Training Accuracy:", decision_train_acc)
print("Decision Tree Testing Accuracy:", decision_test_acc)
print("Confusion Matrix (DT):\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report (DT):\n", classification_report(y_test, y_pred_dt))

# %%
# RANDOM FOREST
# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced', {0: 1, 1: 5}],
}

# RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # Total 20 combinations try karega
    cv=5,  # 5-fold cross-validation karega
    scoring='accuracy',
    n_jobs=-1,  # Parallel computation use karega
    random_state=42
)

# Hyperparameter tuning start
rf_random.fit(X_train, y_train)

# Best parameters print karein
print("Best Parameters:", rf_random.best_params_)

# Best model ko test data par evaluate karein
best_rf_model = rf_random.best_estimator_
y_pred = best_rf_model.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_rf_model_train_acc = accuracy_score(y_train, best_rf_model.predict(X_train))
best_rf_model_test_acc = accuracy_score(y_test, y_pred)
print("Training Accuracy:", best_rf_model_train_acc)
print("Testing Accuracy:", best_rf_model_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %%
# LOGISTIC REGRESSION
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Logistic Regression Model
log_reg = LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)
log_reg.fit(X_train, y_train)

# Predictions on training set
y_pred_train = log_reg.predict(X_train)

# Predictions on test set
y_pred_test = log_reg.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

log_reg_train_acc = accuracy_score(y_train, y_pred_train)
log_reg_test_acc = accuracy_score(y_test, y_pred_test)

# Print results
print("Training Accuracy:", log_reg_train_acc)
print("Testing Accuracy:", log_reg_test_acc)
print("Confusion Matrix on Test Set:\n", confusion_matrix(y_test, y_pred_test))
print("Classification Report on Test Set:\n", classification_report(y_test, y_pred_test))
# %%
# xgboost
# %%
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# XGBoost Classifier with optimized parameters
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=5,
    colsample_bytree=0.8,  # Feature selection better karega
    subsample=0.8,  # Overfitting prevent karega
    gamma=0.2,  # Overfitting ko control karega
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [3, 5, 7]
}

# RandomizedSearchCV for hyperparameter tuning
xgb_random = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,  # 20 combinations try karega
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Train XGBoost with tuning
xgb_random.fit(X_train, y_train)

# Best Model
best_xgb_model = xgb_random.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

xgb_model_train_acc = accuracy_score(y_train, best_xgb_model.predict(X_train))
xgb_model_test_acc = accuracy_score(y_test, y_pred_xgb)
print("Training Accuracy:", xgb_model_train_acc)
print("Testing Accuracy:", xgb_model_test_acc)
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
# %%
# ANN
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Ensure y_train and y_test are numpy arrays
y_train = np.array(y_train).reshape(-1, 1)  # Reshape to (n_samples, 1)
y_test = np.array(y_test).reshape(-1, 1)  # Reshape to (n_samples, 1)

# Convert dataset to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create PyTorch dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define Improved ANN Model
class FraudDetectionANN(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)  # No sigmoid here, BCEWithLogitsLoss will handle it
        return x


# Initialize model, loss function, and optimizer
input_dim = X_train_tensor.shape[1]
model = FraudDetectionANN(input_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005)

# Training loop
epochs = 50
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y.view(-1, 1).float())
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Predictions and evaluation
with torch.no_grad():
    y_pred_test = model(X_test_tensor).squeeze().numpy()
    y_pred_test = [1 if prob > 0 else 0 for prob in y_pred_test]  # No need for sigmoid explicitly

# Accuracy and classification report
ann_accuracy = accuracy_score(y_test, y_pred_test)
print("ANN Model Accuracy:", ann_accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# %%
# Model comparision
# %%
models = pd.DataFrame({
    'Model': ['svc model', 'Decision tree', 'Random forest', 'Logistic Regression', 'XgBoost', 'ANN'],
    'Score': [svc_model_test_acc, decision_test_acc, best_rf_model_test_acc, log_reg_test_acc, xgb_model_test_acc,
              ann_accuracy]
})
models.sort_values(by='Score', ascending=False)
# %%
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Score', data=models, palette='coolwarm')
plt.xlabel('Model Name', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14)
plt.xticks(rotation=45)
plt.show()
# %%
import pickle

# %%
with open('final_random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)
# %%
