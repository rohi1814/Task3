# Task3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('bank-full.csv', sep=';')

# Encode target column
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df.drop('y', axis=1), drop_first=True)

# Feature matrix and target
X = df_encoded
y = df['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.title("Bank Marketing Decision Tree")
plt.show()
