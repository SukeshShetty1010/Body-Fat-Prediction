import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 
import seaborn as sns
from scipy.stats import boxcox

bodyfat = pd.read_csv("D:/ML Atria/bodyfat.csv")

independent_variables = bodyfat.drop(['BodyFat'],axis=1)
dependent_variables = bodyfat['BodyFat']

# --------------------------------- first linear regression ----------------------------------------#

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(independent_variables, dependent_variables)

# Make predictions using the testing set
bodyfat_prediction = regr.predict(independent_variables)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(dependent_variables, bodyfat_prediction))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(dependent_variables, bodyfat_prediction))

# --------------------------------- Cooks distance ----------------------------------------#

def remove_outliers_cooks_distance(bodyfat, target_col):
    independent_variables = bodyfat.drop([target_col], axis=1)
    dependent_variables = bodyfat[target_col]

    def cooks_distance(bodyfat_prediction, bodyfat_prediction_i, m, d):
        cd = np.linalg.norm(bodyfat_prediction-bodyfat_prediction_i)**2/(m*d)
        return cd

    def getCooksDistance(independent_variables, dependent_variables, bodyfat_prediction):
        cooks_dist = []
        mses = []
        independent_variables = np.array(independent_variables)
        dependent_variables = np.array(dependent_variables)
        bodyfat_prediction = np.array(bodyfat_prediction)
        outliers = []
        for i in range(len(bodyfat_prediction)):
            x_i = np.delete(independent_variables, i, axis=0)
            y_i = np.delete(dependent_variables, i, axis=0)
            regr.fit(x_i,y_i)
            bodyfat_prediction_i = regr.predict(x_i)
            mses.append(mean_squared_error(y_i,bodyfat_prediction_i))
            test_bodyfat_prediction = np.delete(bodyfat_prediction,i,axis=0)
            cooks_dist.append(cooks_distance(test_bodyfat_prediction,bodyfat_prediction_i,mses[i],independent_variables.shape[1]))
        return cooks_dist

    independent_variables = np.array(independent_variables)
    dependent_variables = np.array(dependent_variables)
    regr.fit(independent_variables, dependent_variables)
    dependent_variables_pred = regr.predict(independent_variables)

    cooks_distances = getCooksDistance(independent_variables, dependent_variables, dependent_variables_pred)
    outlier_index = np.argmax(cooks_distances) 
    bodyfat_outlier_removed = bodyfat.drop(index=[outlier_index])
    return bodyfat_outlier_removed

bodyfat_clean = remove_outliers_cooks_distance(bodyfat, 'BodyFat')

independent_variables_clean = bodyfat_clean.drop(['BodyFat'], axis=1)
dependent_variables_clean = bodyfat_clean['BodyFat']
regr.fit(independent_variables_clean, dependent_variables_clean)
bodyfat_prediction_clean = regr.predict(independent_variables_clean)
mse_clean = mean_squared_error(dependent_variables_clean, bodyfat_prediction_clean)
r2_clean = r2_score(dependent_variables_clean, bodyfat_prediction_clean)


print("Coefficients(After removing outliners): \n", regr.coef_)
print("Mean squared error(After removing outliners): %.2f", mse_clean)
print("Coefficient of determination(After removing outliners): %.2f", r2_clean)


plt.scatter(dependent_variables[:250], bodyfat_prediction[:250])
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs predicted values for original data')
plt.show()

plt.scatter(dependent_variables_clean[:250], bodyfat_prediction_clean[:250])
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs predicted values for cleaned data')
plt.show()

# --------------------------------- Box Cox Transformation ----------------------------------------#


dependent_variables_clean = np.array(dependent_variables_clean) + 0.01
dependent_variables_boxcox, lambda_boxcox = boxcox(dependent_variables_clean)

# Fit a linear regression model to the transformed data
regr.fit(independent_variables_clean, dependent_variables_boxcox)
bodyfat_prediction_boxcox = regr.predict(independent_variables_clean)

# Inverse transform the predicted output variable to get the original scale
dependent_variables_pred = np.power((bodyfat_prediction_boxcox * lambda_boxcox + 1), 1/lambda_boxcox)

# Check for missing values in dependent_variables and dependent_variables_pred
print('Missing values in dependent_variables:', np.isnan(dependent_variables_clean).sum())
print('Missing values in dependent_variables_pred:', np.isnan(dependent_variables_pred).sum())

# Remove rows with missing values from dependent_variables and dependent_variables_pred
dependent_variables = dependent_variables_clean[~np.isnan(dependent_variables_pred)]
dependent_variables_pred = dependent_variables_pred[~np.isnan(dependent_variables_pred)]

# Calculate MSE and R-squared for the transformed and original data
mse_boxcox = mean_squared_error(dependent_variables_boxcox, bodyfat_prediction_boxcox)
r2_boxcox = r2_score(dependent_variables_boxcox, bodyfat_prediction_boxcox)


# Print the MSE, R-squared, and lambda value
print('MSE (Box-Cox transformed):', mse_boxcox)
print('R-squared (Box-Cox transformed):', r2_boxcox)
print('Lambda value:', lambda_boxcox)

residuals = dependent_variables_clean - bodyfat_prediction_clean
std_residuals = residuals / np.sqrt(mean_squared_error(dependent_variables_clean,bodyfat_prediction_clean))

plt.scatter(bodyfat_prediction_clean, std_residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Standardized Residuals')
plt.show()

plt.scatter(dependent_variables_boxcox[:250], bodyfat_prediction_boxcox[:250])
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs predicted values after BOX COX transformation')
plt.show()
