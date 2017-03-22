# BIA656 Lab 2

# split data into training set and testing set


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


cd = pd.read_csv("./credit-data-post-import.csv")

train_data, test_data = train_test_split(cd, test_size = 0.25)
train_num = pd.value_counts(train_data.monthly_income.isnull())
print(train_num)
test_num = pd.value_counts(test_data.monthly_income.isnull())
print(test_num)




# split data into two groups : nulls and no nulls on the monthly_income variable

train_w_monthly_income = train_data[train_data.monthly_income.isnull() == False]
train_w_null_monthly_income = train_data[train_data.monthly_income.isnull() == True]


# using Random Forest Regression and making predictions

income_imputer = RandomForestRegressor(n_jobs = 1)
cols = [ 'number_real_estate_loans_or_lines' ,  'number_of_open_credit_lines_and_loans']
income_imputer.fit(train_w_monthly_income[cols] , train_w_monthly_income.monthly_income)
new_income_value = income_imputer.predict(train_w_null_monthly_income[cols])

train_w_null_monthly_income.loc[:,('monthly_income')] = new_income_value


# save dataset

train_new_data = train_w_monthly_income.append(train_w_null_monthly_income)

test_data.loc[:,('monthly_income_imputed')] = income_imputer.predict(test_data[cols])
test_data.loc[:,('monthly_income')] = np.where(test_data.monthly_income.isnull(), test_data.monthly_income_imputed, test_data.monthly_income)

train_new_data.to_csv("/Users/admin/Desktop/BIA656/Lab3/credit-data-trainingset.csv", index=False)
test_data.to_csv("/Users/admin/Desktop/BIA656/Lab3/credit-data-testset.csv", index=False)

train_new_num = pd.value_counts(train_new_data.monthly_income.isnull())
test_new_num = pd.value_counts(test_data.monthly_income.isnull())

print(train_new_num)
print(test_new_num)

