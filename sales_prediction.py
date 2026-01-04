# =========================================
# Sales Prediction using Python
# (Auto CSV Creation)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------------------
# 0. Auto-create CSV if not exists
# -----------------------------------------
csv_file = "sales_data.csv"

if not os.path.exists(csv_file):
    print("CSV file not found. Creating sales_data.csv automatically...")

    data = {
        "Advertising_Spend": [
            20000, 25000, 30000, 18000, 22000,
            27000, 32000, 15000, 21000, 26000,
            31000, 19000, 24000, 28000, 33000,
            17000, 23000, 29000, 34000, 36000
        ],
        "Platform": [
            "Online", "TV", "Online", "Print", "Online",
            "TV", "Online", "Print", "TV", "Online",
            "Online", "Print", "TV", "Online", "Online",
            "Print", "TV", "Online", "Online", "Online"
        ],
        "Target_Segment": [
            "Youth", "Adults", "Youth", "Adults", "Youth",
            "Adults", "Youth", "Adults", "Adults", "Youth",
            "Youth", "Adults", "Adults", "Youth", "Youth",
            "Adults", "Adults", "Youth", "Youth", "Youth"
        ],
        "Sales": [
            150000, 180000, 210000, 130000, 160000,
            200000, 230000, 120000, 170000, 195000,
            225000, 140000, 175000, 205000, 240000,
            125000, 165000, 215000, 250000, 265000
        ]
    }

    df_auto = pd.DataFrame(data)
    df_auto.to_csv(csv_file, index=False)

    print("sales_data.csv created successfully!\n")

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------
df = pd.read_csv(csv_file)
print(df.head())

# -----------------------------------------
# 2. Data Cleaning
# -----------------------------------------
df.dropna(inplace=True)

# Encode categorical columns
df['Platform'] = df['Platform'].astype('category').cat.codes
df['Target_Segment'] = df['Target_Segment'].astype('category').cat.codes

# -----------------------------------------
# 3. Feature Selection
# -----------------------------------------
X = df[['Advertising_Spend', 'Platform', 'Target_Segment']]
y = df['Sales']

# -----------------------------------------
# 4. Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------------------------
# 5. Model Training
# -----------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------
# 6. Prediction
# -----------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------
# 7. Evaluation
# -----------------------------------------
print("\nR2 Score:", r2_score(y_test, y_pred))

# -----------------------------------------
# 8. Visualization
# -----------------------------------------
plt.figure()
plt.plot(y_test.values, label="Actual Sales")
plt.plot(y_pred, label="Predicted Sales")
plt.legend()
plt.title("Sales Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.show()

# -----------------------------------------
# 9. Marketing Insight
# -----------------------------------------
print("\nModel Coefficients:")
print("Advertising Spend Impact:", model.coef_[0])
print("Platform Impact:", model.coef_[1])
print("Target Segment Impact:", model.coef_[2])
