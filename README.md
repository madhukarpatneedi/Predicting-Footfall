# Predicting-Footfall
This dataset contains lot of missing values which we have imputed using following approaches:
1. Group by Year, Month, Day, Location_Type and Park_ID and replacing NA with mean of respective columns
2. Group by only Month and replacing NA with mean of respective columns
3. Predicted missing values using KNN
4. Predicted missing values using Random Forest Regressor
5. Predicted missing values using Gradient Boosting Regressor
6. Average of all the ways with which we imputed missing values

Model Fitting:
1. We tried fitting GradientBoostingRegressor,RandomForestRegressor,ExtraTreesRegressor.
2. Finally we got performance by stacking the ensemble of best models of the above models and then fitting it with GradientBoosting Regressor.
