# This is a House price prediction project using different Machine Algorithm used.

## Getting Started

## Description Of Dataset

1. **Transaction Date :** The date in which house is purchased.
2. **House of Age:** How old is the house.
3. **Distance from nearest Metro station (km):** How far from the metro station.
4. **Number of convenience stores:** How many conveinence stores are present.
5. **latitude:** Latitude of the house.
6. **longitude:** Longitude of the house.
7. **Number of bedrooms:** How many Number of bedrooms are in the house.
8. **House size (sqft):** How big or small the house
9. **House price of unit area:** How much price of the house per unit area.

## Exploratory Data Analysis

###### In House price of unit area has some outliers are present and is skewed to the left. so it is necessary to remove these outliers from the dataset.

1. **Remove Outliers.**
2. **Datetime datatypes.**
3. **variable transformation and typecasting.**
4. **Categorical Vs target variables.**
5. **separating the independent variable and target variable.**
6. **Feature scalling.**.
7. **check Multicollinearity and remove.**

## Split the Dataset into train-test-split
  * **split the dataset into 70:30**

## Model Building:-

## 1. Linear Regression :- 
Linear Regression model's main aim is to find the best fit linear line and the optimal values of intercept and coefficients such that the error is minimized. Error is the difference between the actual value and predicted value and the goal is to reduce this difference.
***The Linear Regression score is approx 62.58% and RMSE is 7.03***
  
## 2. GridSearchCv:-
***It is a library function that helps you loop through pre-defined hyperparameters and fits your model with the best ones.***

  **Advantages:** GridSearchCV tries all the combinations of the values passed in the dictionary and evaluates the model for each combination using the Cross-Validation method.  
   **Disadvantages:** It is very time consuming.

## 3. Ridge Regression:-
Ridge Regression (also called Tikhonov regularization) is a regularized version of Linear Regression: a regularization term equal to αΣi = 1n θi2 is added to the cost function. This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible.
   **The Ridge Regression score is approx 62.43% and RMSE is 7.05**
   
## 4. Lasso Regression:-
Least Absolute Shrinkage and Selection Operator Regression (simply called Lasso Regression) is another regularized version of Linear Regression: just like Ridge Regression, it adds a regularization term to the cost function, but it uses the ℓ1 norm of the weight vector instead of half the square of the ℓ2 norm. It is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical model.   
   ***Lasso Regression Score is approx 63% and RMSE is 7.05***
## 5. Elastic Net:-
Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and you can control the mix ratio r. When r = 0, Elastic Net is equivalent to Ridge Regression, and when r = 1, it is equivalent to Lasso Regression. Elastic net is a penalized linear regression model that includes both the L1 and L2 penalties during training.
  ***Elastic Net score is approx 62.75% and RMSE is 7.02***
## 6. Support vector Reggressor:-
Support vector regression is a supervised machine learning algorithm. The basic idea behind SVR is to find the best fit line.It gives us the flexibility to define how much error is acceptable in our model and will find an appropriate line ( or hyperplane in higher dimensions) to fit the data.
    ***Support vector Reggressor score is 73.34% and RMSE is 5.94***
## 7. Decision Tree Regression:-
Decision Tree Regression is a supervised machine learning algorithm. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.
   Decision trees regressors normally use mean squared error (MSE) to decide to split a node in two or more sub-nodes.   
   ***Decision Tree Regressor score is 71.76% and RMSE is 6.11***
## 8. Random Forest Regressor:-
Random Forest Regressor algorithm combines ensemble learning methods with the decision tree framework to create multiple randomly drawn decision trees from the data, averaging the results to output a new result that often leads to strong predictions.
   ***Random Forest Regressor score is 76.34% and RMSE is 5.59***
## 9. AdaBoost Regressor:-
AdaBoost regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases.
   ***AdaBoost Regressor score is 74.90% and RMSE is 5.76*** 
## 10. XGBRegressor:-
XGBBoost Regressor is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. It is a powerful approch for building supervised regression models. The validity of this statement can be inferred by knowing about its(XGBoost) objective function and base learners. The objective function contains loss function and a regularization term. It tells about the differenc between actual values and predicted values, i.e how far the model results are from the real values.
    ***XGBRegressor score is 73.36% and RMSE is 5.93***
    
## Final Result:-
***In this housing price we try different models to check which model has less Root Mean Squared Error, From all of this model's Random Forest Regressor has less RMSE and it gives approx 76% score,therefore Random Forest Regressor Model is best fit for this housing price model.***    


## Conclusion:-
***From this Housing Data Model we conclude that if the House is far away from the Metro station and the House age is old then the price of house is decrease. If the Number of convenience stores are high then the price of house is increase.***
   
  

