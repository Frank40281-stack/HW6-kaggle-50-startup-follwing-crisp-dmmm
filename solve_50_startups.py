import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("=" * 80)
    print("          CRISP-DM METHODOLOGY: 50 STARTUPS PROFIT PREDICTION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1: BUSINESS UNDERSTANDING
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1: BUSINESS UNDERSTANDING ---")
    print("Objective: Predict startup profit based on R&D Spend, Administration Spend,")
    print("           Marketing Spend, and location (State).")
    print("Business Goal: Help venture capitalists and startup founders identify")
    print("               which expenditure types and locations yield highest profits")
    print("               to optimize resource allocation.")

    # -------------------------------------------------------------------------
    # PHASE 2: DATA UNDERSTANDING
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: DATA UNDERSTANDING ---")
    csv_path = "50_Startups.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the download command first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Dataset Loaded. Shape: {df.shape} (50 rows, 5 columns)")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nMissing values count:")
    print(df.isnull().sum())
    
    print("\nUnique States list:")
    print(df['State'].unique())

    # Visualizing relationships and distributions (EDA)
    print("\nGenerating Exploratory Data Analysis (EDA) visualizations...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of Profit
    sns.histplot(df['Profit'], kde=True, ax=axes[0, 0], color='teal')
    axes[0, 0].set_title('Distribution of Profit')
    
    # 2. R&D Spend vs Profit
    sns.scatterplot(data=df, x='R&D Spend', y='Profit', hue='State', s=100, ax=axes[0, 1])
    axes[0, 1].set_title('R&D Spend vs Profit')
    
    # 3. Marketing Spend vs Profit
    sns.scatterplot(data=df, x='Marketing Spend', y='Profit', hue='State', s=100, ax=axes[1, 0])
    axes[1, 0].set_title('Marketing Spend vs Profit')
    
    # 4. Profit by State (Boxplot)
    sns.boxplot(data=df, x='State', y='Profit', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Profit Distribution by State')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=300)
    print("EDA plots saved successfully to 'eda_plots.png'.")
    plt.close()

    # -------------------------------------------------------------------------
    # PHASE 3: DATA PREPARATION
    # -------------------------------------------------------------------------
    print("\n--- PHASE 3: DATA PREPARATION ---")
    # Separating features and target
    X = df.drop(columns=['Profit'])
    y = df['Profit']
    
    print(f"Features columns (X): {list(X.columns)}")
    print(f"Target column (y): {y.name}")

    # Split dataset into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Define preprocessing pipeline for numeric and categorical columns
    numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    categorical_features = ['State']

    # ColumnTransformer to standard scale numbers and one-hot encode state
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    print("Data preprocessor set up: StandardScaler for numeric features, OneHotEncoder for State.")

    # -------------------------------------------------------------------------
    # PHASE 4: MODELING & PHASE 5: EVALUATION
    # -------------------------------------------------------------------------
    print("\n--- PHASE 4 & 5: MODELING & EVALUATION ---")
    
    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        # Build pipeline containing preprocessing and the regressor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'R2 Score': r2,
            'MAE': mae,
            'RMSE': rmse
        })
        print(f"Trained {name:20} -> R2 Score: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # Display comparison
    results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
    print("\nModel Performance Comparison (sorted by R2 Score):")
    print(results_df.to_string(index=False))
    
    # Save comparison to file
    results_df.to_csv("evaluation_metrics.csv", index=False)
    print("Evaluation metrics saved to 'evaluation_metrics.csv'.")

    # Select the best model based on R2 Score
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\nWinner: {best_model_name} is the best performing model!")

    # Plot predictions vs actuals for the best model
    best_y_pred = best_pipeline.predict(X_test)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=best_y_pred, s=100, color='crimson')
    # Perfect fit line
    min_val = min(y_test.min(), best_y_pred.min())
    max_val = max(y_test.max(), best_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='darkblue', label='Perfect Fit')
    plt.title(f'Actual vs. Predicted Profit ({best_model_name})')
    plt.xlabel('Actual Profit')
    plt.ylabel('Predicted Profit')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300)
    print("Model evaluation prediction plot saved to 'model_evaluation.png'.")
    plt.close()

    # -------------------------------------------------------------------------
    # FEATURE SELECTION & PERFORMANCE ANALYSIS (ALL IN ONE)
    # -------------------------------------------------------------------------
    print("\n--- FEATURE SELECTION & PERFORMANCE ANALYSIS (ALL IN ONE) ---")
    
    # We will use the exact train_test_split (test_size=0.2, random_state=0)
    df_encoded = pd.get_dummies(df, columns=['State'], dtype=float)
    y_encoded = df_encoded['Profit']
    
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
        df_encoded, y_encoded, test_size=0.2, random_state=0
    )
    
    # Under the hood feature sets
    feature_sets_actual = [
        ['R&D Spend'],
        ['R&D Spend', 'Marketing Spend'],
        ['R&D Spend', 'Marketing Spend', 'State_New York'],
        ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
        ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
    ]
    
    # Display labels (as shown in the user's target image)
    feature_labels = [
        '[R&D Spend]',
        '[R&D Spend, Marketing Spend]',
        '[R&D Spend, Marketing Spend, State_New York]',
        '[R&D Spend, Marketing Spend, State_New York, State_Florida]',
        '[R&D Spend, Marketing Spend, State_New York, State_Florida, State_California]'
    ]
    
    num_features = [1, 2, 3, 4, 5]
    
    # Define the 5 models
    models_to_compare = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    fs_results = []
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {
        'Linear Regression': 'tab:blue',
        'Ridge Regression': 'tab:orange',
        'Lasso Regression': 'tab:green',
        'Random Forest': 'tab:red',
        'Gradient Boosting': 'tab:purple'
    }
    
    for name, model in models_to_compare.items():
        rmses = []
        r2_scores = []
        
        for idx, f_set in enumerate(feature_sets_actual):
            X_tr = X_train_enc[f_set]
            X_te = X_test_enc[f_set]
            
            model.fit(X_tr, y_train_enc)
            p = model.predict(X_te)
            
            rmse_val = np.sqrt(mean_squared_error(y_test_enc, p))
            r2_val = r2_score(y_test_enc, p)
            
            rmses.append(rmse_val)
            r2_scores.append(r2_val)
            
            fs_results.append({
                'Model': name,
                'Num Features': idx + 1,
                'Features': feature_labels[idx],
                'RMSE': rmse_val,
                'R-squared': r2_val
            })
            
        # Plot curves for each model
        axes[0].plot(num_features, rmses, marker='o', label=name, color=colors[name], linewidth=2)
        axes[1].plot(num_features, r2_scores, marker='o', label=name, color=colors[name], linewidth=2)
        
    # Formatting Plot 1: RMSE
    axes[0].set_title('RMSE by Number of Features (All Models)', fontsize=14)
    axes[0].set_xlabel('Number of Features', fontsize=12)
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_xticks(num_features)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    # Formatting Plot 2: R-squared
    axes[1].set_title('R-squared by Number of Features (All Models)', fontsize=14)
    axes[1].set_xlabel('Number of Features', fontsize=12)
    axes[1].set_ylabel('R-squared', fontsize=12)
    axes[1].set_xticks(num_features)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save the plot as png and then copy it to other requested extensions to avoid matplotlib format errors
    plt.savefig('feature_selection_performance_allinone.png', dpi=300)
    plt.savefig('allinone.png', dpi=300)
    plt.close()
    
    import shutil
    shutil.copyfile('feature_selection_performance_allinone.png', 'feature_selection_performance_allinone.pnd')
    print("All-in-one feature selection performance plot saved to 'feature_selection_performance_allinone.png' & 'feature_selection_performance_allinone.pnd'.")
    
    # Save comparison dataframe to CSV
    fs_results_df = pd.DataFrame(fs_results)
    fs_results_df.to_csv('feature_selection_metrics_all.csv', index=False)

    # -------------------------------------------------------------------------
    # PHASE 6: DEPLOYMENT
    # -------------------------------------------------------------------------
    print("\n--- PHASE 6: DEPLOYMENT ---")
    
    # Save the pipeline
    model_filename = 'best_model.joblib'
    joblib.dump(best_pipeline, model_filename)
    print(f"Best model pipeline (preprocessors + model) successfully saved as '{model_filename}'.")
    
    # Sample prediction demonstration
    print("\nDeployment Test / Making a Sample Prediction:")
    sample_data = pd.DataFrame([{
        'R&D Spend': 100000.0,
        'Administration': 120000.0,
        'Marketing Spend': 250000.0,
        'State': 'California'
    }])
    
    loaded_pipeline = joblib.load(model_filename)
    prediction = loaded_pipeline.predict(sample_data)[0]
    print("Sample Input:")
    print(sample_data.to_string(index=False))
    print(f"Predicted Profit: ${prediction:,.2f}")
    
    print("\n" + "=" * 80)
    print("                CRISP-DM PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
