import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the dataset
df = pd.read_csv('brca.csv')

# Handle Missing Values
df = df.dropna()

# Encode Categorical Variables using Label Encoding
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Scale Numerical Features
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'y' in numerical_cols:
    numerical_cols.remove('y')
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split into Train/Test Sets (80/20)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model (as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# You would typically add code here to load the model and make predictions
# For example:
# with open('model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)
# predictions = loaded_model.predict(new_data)
