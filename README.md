# Body-Fat-Prediction
 This project analyzes body fat data using regression modeling. It includes linear regression, outlier detection (Cook's distance), and Box-Cox transformation. It provides insights, improves modeling accuracy, and predicts body fat. A concise and comprehensive analysis of body fat, including outlier handling and transformation techniques.

The project involves performing data analysis and regression modeling on a dataset containing body fat measurements. The code utilizes various techniques such as linear regression, outlier detection using Cook's distance, and Box-Cox transformation.

First, a linear regression model is trained on the original data to predict body fat measurements. The model's coefficients, mean squared error, and coefficient of determination (R-squared) are calculated and printed.

Next, Cook's distance is employed to identify and remove outliers from the dataset. A modified dataset without outliers is created, and the linear regression model is retrained on this cleaned data. The updated model's coefficients, mean squared error, and R-squared are displayed.

Afterward, the Box-Cox transformation is applied to the dependent variable (body fat measurements) to achieve a more normal distribution. The transformed data is used to train another linear regression model, and the predictions are inverse-transformed to obtain results on the original scale. The mean squared error, R-squared, and lambda value (parameter of the Box-Cox transformation) are printed.

Lastly, the standardized residuals are calculated and plotted to assess the model's performance. Additionally, a scatter plot comparing the true and predicted values after the Box-Cox transformation is displayed.

Overall, the project demonstrates a comprehensive analysis of the body fat dataset, including regression modeling, outlier detection, and transformation techniques, providing valuable insights into the relationship between independent variables and body fat measurements.
