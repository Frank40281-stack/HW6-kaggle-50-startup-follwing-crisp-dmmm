# 🚀 50 Startups Profit Prediction System (CRISP-DM Framework)

This repository contains the complete implementation of the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology for predicting startup profits based on the 50 Startups dataset. It evaluates multiple machine learning models and performs Sequential Forward Feature Selection to find the optimal features.

---

## 📅 Project Structure & Direct Links
*   **Dataset**: [50_Startups.csv](file:///d:/曾昱誠AI作業/hw6-0609/50_Startups.csv)
*   **Machine Learning Script**: [solve_50_startups.py](file:///d:/曾昱誠AI作業/hw6-0609/solve_50_startups.py)
*   **Interactive Web App**: [app.py](file:///d:/曾昱誠AI作業/hw6-0609/app.py)
*   **Walkthrough Notes**: [walkthrough.md](file:///C:/Users/admin/.gemini/antigravity-ide/brain/c425c993-186d-400f-976d-054a757bdef7/walkthrough.md)
*   **Detailed Homework Report**: [hw6.md](file:///d:/曾昱誠AI作業/hw6-0609/hw6.md)

---

## 📝 CRISP-DM Project Summary

### 1. Business Understanding
Predicting the potential profitability of new startups helps venture capitalists (VCs) and founders optimize resource allocation and pick high-yield investment options.

### 2. Data Understanding
The dataset contains 50 startups with features: R&D Spend, Administration, Marketing Spend, State, and target variable Profit. Exploratory Data Analysis (EDA) shows that R&D Spend has a very strong linear correlation with Profit.

![Exploratory Data Analysis](file:///d:/曾昱誠AI作業/hw6-0609/eda_plots.png)

### 3. Data Preparation
- Dataset Split: **80% Training Set (40 samples)** and **20% Test Set (10 samples)**.
- Preprocessing Pipeline:
  - StandardScaler for numerical columns (`R&D Spend`, `Administration`, `Marketing Spend`).
  - OneHotEncoder for the categorical column (`State`).

### 4. Modeling & 5. Evaluation
We evaluated 5 algorithms on the unseen test set:
1.  **Random Forest**: **R² = 0.9277** | MAE = $5,637.57 | RMSE = 7,650.34 (Best Performing Model 🏆)
2.  **Gradient Boosting**: R² = 0.9057 | MAE = $8,139.54 | RMSE = 8,739.75
3.  **Lasso Regression**: R² = 0.8988 | MAE = $6,961.14 | RMSE = 9,054.83
4.  **Linear Regression**: R² = 0.8987 | MAE = $6,961.48 | RMSE = 9,055.96
5.  **Ridge Regression**: R² = 0.8954 | MAE = $7,412.45 | RMSE = 9,203.48

![Actual vs. Predicted Profit](file:///d:/曾昱誠AI作業/hw6-0609/model_evaluation.png)

---

## 📊 Feature Selection Performance (All-in-One)
We performed **Sequential Forward Feature Selection (SFS)** to identify the optimal subset of features for all 5 models:

![All Models SFS Performance](file:///d:/曾昱誠AI作業/hw6-0609/feature_selection_performance_allinone.png)

### 📝 SFS Performance Table (Linear Regression Example)
| Features Count | Selected Subset | RMSE | R-squared |
|:---:|:---|:---:|:---:|
| 1 | `[R&D Spend]` | 8,274.87 | 0.9465 |
| 2 | `[R&D Spend, Marketing Spend]` | **8,198.80** (Optimal 🌟) | **0.9474** (Highest 🌟) |
| 3 | `[R&D Spend, Marketing Spend, State_New York]` | 8,309.06 | 0.9460 |
| 4 | `[R&D Spend, Marketing Spend, State_New York, State_Florida]` | 8,409.92 | 0.9447 |
| 5 | `[R&D Spend, Marketing Spend, State_New York, State_Florida, State_California]` | 9,137.99 | 0.9347 |

*   **Key Finding**: A simplified **2-feature model (R&D Spend + Marketing Spend)** yields the best RMSE and R² score across all models. Adding extra variables like geographical locations or administration expenses introduces noise and multicollinearity, leading to metric degradation (RMSE increases, R² drops).

---

### 6. Deployment
- The Random Forest Regressor pipeline is saved as `best_model.joblib`.
- An interactive **Streamlit dashboard** (`app.py`) is deployed. It features real-time prediction sliders, model performance comparison tables, SFS results filtering dropdowns, and SFS performance line plots.

---

## 🛠️ How to Run

1.  **Install dependencies**:
    ```bash
    pip install matplotlib seaborn scikit-learn pandas numpy joblib streamlit
    ```
2.  **Train models and generate plots**:
    ```bash
    python solve_50_startups.py
    ```
3.  **Run Streamlit dashboard**:
    ```bash
    streamlit run app.py
    ```
