import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import regex as re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,  precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('dataSet_Badrinath.csv')

data

df_encoded = pd.concat([data, pd.get_dummies(data['Type'], prefix='Type').astype(int)], axis=1)
df_encoded = df_encoded.drop(columns=['Type'])
print(df_encoded)

df_encoded['Power'] = (df_encoded['Rotational speed [rpm]'] * (2 * np.pi / 60)) / df_encoded['Torque [Nm]']
df_encoded['Temp_ratio'] = (df_encoded['Process temperature [K]']) / df_encoded['Air temperature [K]']

df_encoded_relevant = df_encoded.drop(columns=['Product ID', 'UDI'])
df_encoded_relevant

X = df_encoded_relevant.drop(['Failure Type','Target'], axis=1)
y = df_encoded_relevant['Target']

#Replacing non-alpha numeric characters with '_' in column names
X.columns = [re.sub(r"[^a-zA-Z0-9_]+", "_", col) for col in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
lr = LogisticRegression()

# Fit the model to the training data
lr.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_lr = lr.predict(X_test)

# Create a confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Plot the confusion matrix using seaborn's heatmap
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Display the plot
plt.show()

y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)

# Print the results
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1 Score: {f1_lr:.4f}")
print(f"AUC: {roc_auc_lr:.4f}")