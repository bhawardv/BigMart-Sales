# BigMart-Sales
Objective: Predict Item_Outlet_Sales using historical sales data from BigMart.
1. Data Understanding & Preprocessing
Data Loading & Cleaning:
•	Standardized categorical values (e.g., Item_Fat_Content: "LF" → "Low Fat").
•	Engineered Outlet_Age from Outlet_Establishment_Year to capture outlet maturity.
•	Imputed zero Item_Visibility with the mean of non-zero values.
•	Dropped redundant columns (Item_Identifier, Outlet_Establishment_Year).
Feature Engineering:
•	Segmented features into numeric (Item_Weight, Item_MRP, etc.) and categorical variables.
•	Used TargetEncoder for high-cardinality categorical features (e.g., Item_Type) to avoid one-hot explosion.
2. Model Architecture
Ensemble Learning:
•	Implemented VotingRegressor with RandomForest, GradientBoosting, and XGBRegressor to combine strengths of diverse algorithms.
•	Rationale: Ensembles reduce overfitting and leverage complementary learning patterns.
Pipeline Design:
•	Integrated cleaning, preprocessing, and modeling into a single Pipeline for reproducibility.
•	Applied StandardScaler to numeric features and OneHotEncoder to low-cardinality categorical features.
3. Model Optimization
Hyperparameter Tuning:
•	Used GridSearchCV to optimize parameters (e.g., n_estimators, max_depth) for each sub-model.
•	Focused on critical parameters to balance performance and computational cost.
Validation:
•	Evaluated via 5-fold cross-validation to ensure robustness.
•	Metrics: RMSE (primary), R², and adjusted R² (to account for feature count).
4. Results & Insights
Performance:
•	Test RMSE: [Value] (lower is better).
•	R²: [Value] (explained variance).
•	Residual analysis showed near-normal distribution, indicating good fit.
Deployment:
•	Saved the model using pickle for future inference.
•	Predicted sales for unseen test data, clipping negative values to zero.
Key Experimentation Steps
•	Feature Engineering: Tested impact of Outlet_Age and Item_Visibility imputation.
•	Encoding Strategies: Compared TargetEncoder vs. OneHotEncoder for high-cardinality features.
•	Model Selection: Evaluated standalone models (e.g., XGBoost) before adopting an ensemble.
•	Hyperparameter Sensitivity: Observed performance changes with learning_rate and max_depth.

Conclusion: The pipeline balances preprocessing rigor, model diversity, and validation, achieving robust sales predictions. Further gains could arise from deeper hyperparameter tuning and advanced feature engineering.
