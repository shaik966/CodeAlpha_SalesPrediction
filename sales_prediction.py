# ==============================================================
#            CODEALPHA INTERNSHIP PROJECT
#               SALES PREDICTION USING PYTHON
#                Developed by : Shaik Numaan
# ==============================================================
# ================= IMPORT LIBRARIES =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# ================= LOAD DATASET =================
sales_df = pd.read_csv("Advertising.csv")
print("\n================================================")
print("         SALES DATASET PREVIEW")
print("================================================\n")
print(sales_df.head())
print("\n================================================")
print("           DATASET INFORMATION")
print("================================================\n")
print(sales_df.info())
print("\n================================================")
print("            RANDOM SAMPLE DATA")
print("================================================\n")
print(sales_df.sample(5))
# ================= REMOVE EXTRA COLUMN =================
if "Unnamed: 0" in sales_df.columns:
    sales_df.drop("Unnamed: 0", axis=1, inplace=True)
# ================= CHECK NULL VALUES =================
print("\n================================================")
print("             MISSING VALUES")
print("================================================\n")
print(sales_df.isnull().sum())
# ================= FEATURES & TARGET =================
X = sales_df[["TV", "Radio", "Newspaper"]]
y = sales_df["Sales"]
# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=101
)
print("\n================================================")
print("             DATASET SHAPES")
print("================================================\n")
print("Training Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)
# ================= LINEAR REGRESSION MODEL =================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
# ================= RANDOM FOREST MODEL =================
forest_model = RandomForestRegressor(
    n_estimators=300,
    random_state=101
)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
# ================= MODEL PERFORMANCE =================
lr_mae = mean_absolute_error(y_test, linear_predictions)
lr_r2 = r2_score(y_test, linear_predictions)
rf_mae = mean_absolute_error(y_test, forest_predictions)
rf_r2 = r2_score(y_test, forest_predictions)
print("\n================================================")
print("         LINEAR REGRESSION RESULTS")
print("================================================\n")
print("Mean Absolute Error :", lr_mae)
print("R2 Score            :", lr_r2)
print("\n================================================")
print("          RANDOM FOREST RESULTS")
print("================================================\n")
print("Mean Absolute Error :", rf_mae)
print("R2 Score            :", rf_r2)
# ================= BEST MODEL =================
if rf_r2 > lr_r2:
    print("\n================================================")
    print(" BEST MODEL : RANDOM FOREST REGRESSOR")
    print("================================================")
else:
    print("\n================================================")
    print(" BEST MODEL : LINEAR REGRESSION")
    print("================================================")
# ==============================================================
# DIAGRAM 1 : ACTUAL VS PREDICTED SALES
# ==============================================================
plt.figure(figsize=(9,6))
plt.scatter(
    y_test,
    forest_predictions,
    s=100,
    alpha=0.8
)
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.title("Actual vs Predicted Sales Analysis", fontsize=15)
plt.grid(True)
plt.show()
# ==============================================================
# DIAGRAM 2 : CORRELATION HEATMAP
# ==============================================================
plt.figure(figsize=(8,6))
sns.heatmap(
    sales_df.corr(),
    annot=True,
    linewidths=1
)
plt.title("Advertising Dataset Correlation Heatmap", fontsize=14)
plt.show()
# ==============================================================
# DIAGRAM 3 : TV ADVERTISING ANALYSIS
# ==============================================================

top_tv = sales_df.sort_values(by="TV", ascending=False).head(10)

plt.figure(figsize=(10,6))

plt.bar(
    top_tv.index,
    top_tv["TV"]
)
plt.xlabel("Top Records", fontsize=12)
plt.ylabel("TV Advertising Budget", fontsize=12)
plt.title("Top 10 TV Advertising Budgets", fontsize=15)
plt.grid(True)
plt.show()
# ==============================================================
# DIAGRAM 4 : SALES DISTRIBUTION
# ==============================================================
plt.figure(figsize=(10,6))
plt.hist(
    sales_df["Sales"],
    bins=10
)
plt.xlabel("Sales", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Sales Distribution Analysis", fontsize=15)
plt.grid(True)
plt.show()
# ==============================================================
# FINAL MESSAGE
# ==============================================================
print("\n================================================")
print("   PROJECT EXECUTED SUCCESSFULLY")
print("   SALES PREDICTION ANALYSIS COMPLETED")
print("================================================")