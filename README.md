# Mini-Project

## PROBLEM STATEMENT
To Predict the House price in future using Data Science and Machine Learning model.

## STEPS
1.First Step is to collect data from dataset.

2.Then some basic Data Science process are to be applied.

3.Then the model is trained using Machine Learning algorithm which in the case using Regression.

4.Based on the generated graphs we predict the cost of the house.

## CODE
```
import pandas as pd
data = pd.read_csv('/content/house-prices.csv')
data.head()
```
![image](https://github.com/ManiKandan228/Mini-Project/assets/119160414/a6d77ffc-1e1c-4f17-9a04-e0556d41bbe6)

```
data.isnull().sum()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
# Separate the features and target variable
X = data.drop('Price', axis=1)  # Features
y = data['Price']  # Target variable

# Perform one-hot encoding for categorical variables
categorical_cols = ['Brick', 'Neighborhood']  # Update with your categorical column names
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Plotting the predicted and actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.show()
```
## OUTPUT

![image](https://github.com/ManiKandan228/Mini-Project/assets/119160414/2acdcc36-d48a-4683-b5aa-4ff6a49f34fe)

## RESULT
From the training and testing of dataset on the model, a strong deduction can be made that on using Data Science model and ML model we can get the better accuracy in the House Price Prediction.
