import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the data
housing = pd.read_csv('housing.csv')

# 2. Create a stratified test set based on income category
housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

# 4. Separate numerical and categorical columns
num_attributes = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attributes = ['ocean_proximity']

# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared.shape)

# 7. Train the model

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print(f'The root mean squared error for Linear Regression is {lin_rmse}')
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f'The root mean squared error for Linear Regression is {dec_rmse}')
print(pd.Series(lin_rmses).describe())

# Decision Tree
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f'The root mean squared error for Linear Regression is {dec_rmse}')
print(pd.Series(dec_rmses).describe())

# Randome Forest Regressor
ranfor_reg = RandomForestRegressor(random_state=42)
ranfor_reg.fit(housing_prepared, housing_labels)
ranfor_preds = ranfor_reg.predict(housing_prepared)
# ranfor_rmse = root_mean_squared_error(housing_labels, ranfor_preds)
# print(f'The root mean squared error for Randome Forest Regression is {ranfor_rmse}')
ranfor_rmses = -cross_val_score(ranfor_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f'The root mean squared error for Linear Regression is {dec_rmse}')
print(pd.Series(ranfor_rmses).describe())