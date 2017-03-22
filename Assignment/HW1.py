#Hwk 1

# Basic data manipulation

from statistics import mode
import pandas as pd

income = pd.read_csv("http://personal.stevens.edu/~gcreamer/BIA656/income1.data")
new_income = income.to_csv("/Users/admin/Desktop/income.csv", index=False)

#Question 1 : the number of lines in the file
file_1 = open("/Users/admin/Desktop/BIA656/income.csv")
lines = file_1.readlines()

print("Q1 : the number of lines : %d" % int(len(lines)))


#Question 2

file_2 = open("/Users/admin/Desktop/BIA656/income.csv")
new_file = open("/Users/admin/Desktop/income1.csv", 'w')

for line in file_2:
	if 'NA' not in line:
		new_file.write(line)

new_file = open("/Users/admin/Desktop/income1.csv")
count2 = sum(1 for line in new_file)
print("Q2 : the number of lines : %d" % count2)

#Q3: Indicate the most common education level (the fifth column corresponds to education level).
file_3 = pd.read_csv("http://personal.stevens.edu/~gcreamer/BIA656/income1.data", delimiter= '\s+',header=None,prefix='X')

for value in file_3['X4']:
	lst = []
	index = 0
	lst.append(value)
	index += 1

column = lst
column_mo = mode(column)

print("Q3 : most common value : %d" % int(column_mo))


#Q4 : Indicate the level of income for households with some graduate school.
file_4= pd.read_csv('http://personal.stevens.edu/~gcreamer/BIA656/income1.data',delimiter= '\s+',header=None,prefix='X')
income_level = file_4.X0[file_4.X4 == 6].tolist()
print(file_4.head())
print(file_4.describe())

categories = ['Less than $10,000','$10,000 to $14,999','$15,000 to $19,999','$20,000 to $24,999','$25,000 to $29,999',
        '$30,000 to $39,999','$40,000 to $49,999','$50,000 to $74,999','$75,000 or more']

for x in categories:
	print(x,income_level.count(categories.index(x)+1))


 