#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# 1. Importing necessary libraries and basic exploratory data analysis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce  

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Load the train dataset
df = pd.read_csv('train.csv')


# In[3]:


df.head(5)


# In[4]:


df.info()


# In[ ]:





# In[5]:


# =============================================================================
# 2. Separate Features and Target
# =============================================================================
X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']


# In[6]:


# =============================================================================
# 3. Data Cleaning using FunctionTransformer
#    - Standardize 'Item_Fat_Content'
#    - Create 'Outlet_Age' = 2013 - Outlet_Establishment_Year
#    - Replace zero 'Item_Visibility' with mean of non-zero values
#    - Drop unneeded columns ('Outlet_Establishment_Year' and 'Item_Identifier')
# =============================================================================
def clean_data(df):
    # Standardize 'Item_Fat_Content'
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'reg': 'Regular'
    })
    
    # Create 'Outlet_Age'
    df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
    
    # Replace zero 'Item_Visibility' with the mean of non-zero values
    df['Item_Visibility'] = df['Item_Visibility'].mask(df['Item_Visibility'] == 0, 
                                                      df.loc[df['Item_Visibility'] > 0, 'Item_Visibility'].mean())
    
    # Drop unneeded columns
    df = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier'])
    
    return df


clean_transformer = FunctionTransformer(clean_data, validate=False)


# In[7]:


# =============================================================================
# 4. Define Numeric and Categorical Columns (after cleaning)
#    Numeric: impute and scale.
#    Categorical: impute and one-hot encode.
# =============================================================================
numeric_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
low_card_cols = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
high_card_cols = ['Item_Type', 'Outlet_Identifier']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

low_card_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

high_card_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_enc', ce.TargetEncoder())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('low_cat', low_card_pipeline, low_card_cols),
    ('high_cat', high_card_pipeline, high_card_cols)
])


# In[21]:


# =============================================================================
# 5. Build Final Pipeline with VotingRegressor
# =============================================================================
model_pipeline = Pipeline([
    ('clean', clean_transformer),
    ('preprocess', preprocessor),
    ('model', VotingRegressor([
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor()),
                
    ]))
])


# In[22]:


# =============================================================================
# 6. Split Data into Training and Test Sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


# =============================================================================
# 7. Check Variance Inflation Factor (VIF) for Numeric Features
#    We apply the cleaning and numeric pipeline on training numeric data.
# =============================================================================
X_train_clean = clean_transformer.transform(X_train)
numeric_data = X_train_clean[numeric_cols]

# Impute and scale numeric data (and convert back to DataFrame for VIF)
numeric_data_transformed = numeric_pipeline.fit_transform(numeric_data)
numeric_data_df = pd.DataFrame(numeric_data_transformed, columns=numeric_cols, index=numeric_data.index)

vif_data = pd.DataFrame()
vif_data["feature"] = numeric_data_df.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_data_df.values, i) for i in range(numeric_data_df.shape[1])]
print("VIF for numeric features:")
print(vif_data)


# In[32]:


# =============================================================================
# 8. Hyperparameter Tuning with GridSearchCV
#    We tune parameters of the sub-estimators within the VotingRegressor.
# =============================================================================
param_grid = {
    'model__gb__n_estimators':[100],   #GradientBoosting parameters
    'model__gb__learning_rate':[0.05],
    'model__gb__min_samples_split':[10],
    'model__gb__min_samples_leaf':[1],
    'model__gb__max_depth':[3],
    
    'model__rf__n_estimators': [200],  # RandomForest parameters
    'model__rf__max_depth': [6],
    'model__rf__min_samples_split': [1],
    'model__rf__min_samples_leaf': [8],
    'model__rf__max_features': [0.8],
    
    'model__xgb__n_estimators': [180],  # XGBoost parameters
    'model__xgb__learning_rate': [0.03],
    'model__xgb__max_depth': [3],
    'model__xgb__min_child_weight': [1],
    
    
       
    
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=model_pipeline,  # Your pipeline
    param_grid=param_grid,     # Parameter grid
    cv=5,                      # 5-fold cross-validation
    scoring='neg_mean_squared_error',  # Scoring metric
    n_jobs=-1                  
)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and score
print("\nBest Hyperparameters:")
print(grid_search.best_params_)
print("Best Score (neg MSE):", grid_search.best_score_)


# In[33]:


# =============================================================================
# 9. Cross-Validation on the Best Estimator
# =============================================================================
cv_results = cross_validate(grid_search.best_estimator_, X_train, y_train, cv=5, 
                            scoring=('neg_mean_squared_error', 'r2'), return_train_score=True)
print("\nCross-Validation Results:")
for key, value in cv_results.items():
    print(f"{key}: {value}")


# In[34]:


# =============================================================================
# 10. Evaluate the Final Model on Test Set
# =============================================================================
y_pred = grid_search.best_estimator_.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

n = len(y_test)        # Number of samples in the test set
p = X_test.shape[1]    # Number of features (raw input features)
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nTest Set Evaluation:")
print("RMSE:", rmse)
print("R^2:", r2)
print("Adjusted R^2:", adjusted_r2)


# In[35]:


residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# In[ ]:





# In[36]:


import pickle

# Save the model to a file
with open('sales_prediction_model.pkl', 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)


# In[37]:


# Load the model from the file
with open('sales_prediction_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[38]:


# Load the testing data
test_df = pd.read_csv('test.csv')

# Now use the loaded model to predict on test data
y_pred_loaded = loaded_model.predict(test_df)


# In[39]:


test_df['Item_Outlet_Sales']= y_pred_loaded


# In[40]:


print(test_df['Item_Outlet_Sales'].head(10))


# In[41]:


test_df = test_df[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]


# In[42]:


test_df['Item_Outlet_Sales'] = np.where(test_df['Item_Outlet_Sales']<0,0,test_df['Item_Outlet_Sales'])


# In[43]:


# Save the DataFrame with predictions to a new Excel file
test_df.to_csv('Sales_predictions.csv', index=False)


# In[ ]:




