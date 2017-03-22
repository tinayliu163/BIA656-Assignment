# BIA656 Lab 1

import pandas as pd
import pylab as pl
import numpy as np
import re
import os

#Modify the path:
path = "/Users/admin/Desktop/BIA656"
os.chdir(path)

df = pd.read_csv("./credit-training.csv")

# Question 1 : generate a table with number of null values by variable 

num_null = df.isnull().sum()
print(num_null)

# Question 2: convert column names from camelCase into sake_case


column_list = list(df.columns.values)
def convert(name):
	snake_case = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
	return re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_case).lower()

snake_case_list = []
for i in column_list:
	CamelToSnake = convert(i)
	snake_case_list.append(CamelToSnake)

print(snake_case_list)  

# Question 3 

# count the number of cases with the following characteristics

# people 35 or older
age_col = df['age'] >= 35 
num_age_col = age_col.value_counts(sort = True)
print(num_age_col)

# who have not been delinquent in the past 2 years
Dlqin2yrs_not = df.loc[df['SeriousDlqin2yrs'] == 0] 
length =len(Dlqin2yrs_not.index)
print(length)

# who have less than 10 open credit lines/loans
credit= df['NumberOfOpenCreditLinesAndLoans'] < 10
num_credit_col = credit.value_counts(sort = True)
print(num_credit_col)

# combination of these characteristics

Task3 = df.loc[(df['age'] >= 35) & (df['SeriousDlqin2yrs'] == 0) & (df['NumberOfOpenCreditLinesAndLoans'] < 10)] 
length_task3 = len(Task3.index)

print("********************* Question3 ***************************")
print(Task3)
print("Q3 : The number of people : %d" % length_task3)


# Question 4


# people who have been delinquent in the past 2 years

Dlqin2yrs_true = df.loc[df['SeriousDlqin2yrs'] == 1]
print(Dlqin2yrs_true)


# are in the 90th percentile for monthly_income

df_new = df.dropna(axis = 0, how = 'any')

income_col = df_new['MonthlyIncome'].values
nintyth = np.percentile(income_col , 90)
df.loc[df['MonthlyIncome'] == nintyth]

# combination of these characteristics

Task4 = df.loc[(df['SeriousDlqin2yrs'] == 1) & (df['MonthlyIncome'] == nintyth)]

length_task4 = len(Task4.index)

print("********************* Question4 ***************************")
print(Task4)
print("Q4 : The number of people : %d" % length_task4)










