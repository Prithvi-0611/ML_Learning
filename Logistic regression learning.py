import pandas as pd
import numpy as np

df=pd.read_csv('counterfeit_products.csv')
pd.set_option("display.max_columns", None)


df=df.drop(df.columns[[0,1,2,3,4,16,17,23]],axis=1)
df["contact_info_complete"] = df["contact_info_complete"].astype(int)
df["return_policy_clear"]=df["return_policy_clear"].astype(int)
df["return_policy_clear"] = df["return_policy_clear"].astype(int)
df["ip_location_mismatch"] = df["ip_location_mismatch"].astype(int)
df["is_counterfeit"] = df["is_counterfeit"].astype(int)

numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
exclude_cols = ["is_counterfeit", "contact_info_complete", "return_policy_clear", "ip_location_mismatch","unusual_payment_patterns"]
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Standardize all continuous numeric features
df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
df[exclude_cols]=df[exclude_cols].astype(int)

df=df.dropna()
df=df.drop(columns=["listing_date"])

X = df.drop("is_counterfeit", axis=1).values
y = df["is_counterfeit"].values.reshape(-1, 1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize
m, n = X.shape
w = np.zeros((n, 1))
b = 0
alpha = 0.01
epochs = 1000
# Gradient descent
for epoch in range(epochs):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    # Gradients
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    db = (1 / m) * np.sum(y_hat - y)
   # Update
    w -= alpha * dw
    b -= alpha * db
# Final accuracy
final_pred = (sigmoid(np.dot(X, w) + b) >= 0.9).astype(int)
print("Final Training Accuracy:", np.mean(final_pred == y))
