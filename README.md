# Simple Linear Regression
## Linear Regression Model with R:
In this chapter we will start from the linear regression model and then pass to non linear models. I will compare both R and Python along this code.
I want to share with you my notes on Machine Learning so if you are a beginner in machine learning you might find it usefull too.
Here is the simple linear regression with simple dataset composed by 2 variables. Salary and GPA. The model is supervised model so the parameter is already given . The target is to predict the salary with the regressor/covariate GPA.
1. How to split a dataset into training and test datasets.
2. The the summary of Regression output where the null hypothesis and the coefficents are discussed.
3. Predicting the test data in base of the training data.
4. You will find also the definition of MSE,RMSE, R Square and Residuals and their use to evaluate the accuracy of the Regression.
**Here is the code**
```
head(data)
```
![image](https://user-images.githubusercontent.com/90762709/136465743-11ccdaef-b814-4838-99cd-1fb1ba757ebf.png)

```
#Splitting dataset into train (80%) and test(20%) to find an accuracy of the model.
set.seed(1)
train_index = sample(1:nrow(data),0.8*nrow(data),replace = FALSE)

train_data= data[train_index,]
test_data=data[-train_index,]
reg_lm = lm(Salary ~ GPA, data=train_data)
summary(reg_lm)
```
Call: <br />
lm(formula = Salary ~ GPA, data = train_data)  <br />  <br />

Residuals:  <br />
     Min       1Q   Median       3Q      Max  <br />
-225.552  -45.762    5.324   47.646  160.670  <br />

Coefficients: <br /> <br />
            Estimate Std. Error t value Pr(>|t|)     <br />
(Intercept)   981.34     129.67   7.568 1.71e-10 *** <br />
GPA           256.79      38.44   6.681 6.33e-09 *** <br />
--- <br />
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1 <br />

Residual standard error: 80.03 on 65 degrees of freedom <br />
Multiple R-squared:  0.4071,	Adjusted R-squared:  0.398  <br />
F-statistic: 44.63 on 1 and 65 DF,  p-value: 6.332e-09 <br />
```
train_data %>% 
  ggplot(aes(GPA, Salary)) +
  geom_point() +
  geom_smooth(method = "lm")
```
![image](https://user-images.githubusercontent.com/90762709/136466017-3f9fa042-eff5-47e0-9141-005c381f36d8.png) <br />
```
lm_pred = reg_lm %>% predict(test_data)
#Model Performace with Mean Square Errors and Correlation Rate.

lm_perf = 
  data.frame(
    RMSE = sqrt(mean((lm_pred - test_data$Salary)^2)),
    COR = cor(lm_pred, test_data$Salary)
)
lm_perf
```
![image](https://user-images.githubusercontent.com/90762709/136466240-6db709ff-2093-46b9-824f-24361ed4893a.png) <br />

In regression model, the most commonly known evaluation metrics include:<br />

R-squared (R2), which is the proportion of variation in the outcome that is explained by the predictor variables. In multiple regression models, R2 corresponds to the squared correlation between the observed outcome values and the predicted values by the model. The Higher the R-squared, the better the model.<br />

Root Mean Squared Error (RMSE), which measures the average error performed by the model in predicting the outcome for an observation. Mathematically, the RMSE is the square root of the mean squared error (MSE), which is the average squared difference between the observed actual outome values and the values predicted by the model. So, MSE = mean((observeds - predicteds)^2) and RMSE = sqrt(MSE). The lower the RMSE, the better the model. <br />

Residual Standard Error (RSE), also known as the model sigma, is a variant of the RMSE adjusted for the number of predictors in the model. The lower the RSE, the better the model. In practice, the difference between RMSE and RSE is very small, particularly for large multivariate data. <br />

Mean Absolute Error (MAE), like the RMSE, the MAE measures the prediction error. Mathematically, it is the average absolute difference between observed and predicted outcomes, MAE = mean(abs(observeds - predicteds)). MAE is less sensitive to outliers compared to RMSE. (Source: http://www.sthda.com/english/articles/38-regression-model-validation/158-regression-model-accuracy-metrics-r-square-aic-bic-cp-and-more/)<br />

**Let's now predict a random GPA and find what is going to be a salary**
```
salary= reg_lm %>% predict(tibble(GPA = 3.45,1))
salary

```
Output: 1867.255 <br />
**Conclusion with 0.72 Correlation rate the salary of a person who has graduates with 3.45 will be 1867.255.**




