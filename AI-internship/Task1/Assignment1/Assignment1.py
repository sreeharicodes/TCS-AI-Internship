import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1. Using Pandas, load the dataset from https://www.kaggle.com/uciml/iris/data to a variable name iris
iris = pd.read_csv('iris.csv')

#2. Create a list named headers with all the column header names in the given order.
headers = list(iris)

#3. Using the slice operation on headers, extract the column names with index 1 to 4 onto a list called features.
features = headers[1:5]

#4. Display the first five records of iris
print(iris.head())

#5. Make a scatterplot of the Iris features.
iris.plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm')
plt.show()
iris.plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm')
plt.show()

#6. What is the range of ‘SepalLengthCm’ in the dataset? What is the second largest value of ‘SepalLengthCm’ in the dataset?
print('{}-{}'.format(iris['SepalLengthCm'].min(),iris['SepalLengthCm'].max()))
max1 = iris['SepalLengthCm'].max()
l = list(iris['SepalLengthCm'])
l.sort(reverse=True)
for i in l:
    if i != max1:
        print(i)
        break


#7. Find the mean of all the values in SepalWidthCm using numpy
mean = np.array(iris['SepalWidthCm']).mean()
#print(mean)

#8. Identify  ‘SepalLengthCm’  values less than 5. Create a new column named ‘Length’ , categorise each entry as ‘Small’ or ‘Large’, if less than 5.
def create_len(cols):
    len = cols[0]
    if len<5:
        return 'small'
    else:
        return 'large'
iris['Length']=iris[['SepalLengthCm']].apply(create_len,axis=1)
print(iris.head())

#9. Group dataFrame by the "Species" column. Make a histogram of the same.
sp = iris.groupby('Species')
plt.figure();
sp.plot.hist(alpha=0.5,bins=20)
plt.show()

#10 .Find the deviation of length for ‘SepalLengthCm’ from the average
deav = iris['SepalLengthCm'].std()
print(deav)

#11 . Find correlation between columns and display columns with more than 70% percent correlation (either positive or negative).
cor=iris.corr()
c = []

for i in features:
    for j in features:
        if (abs(cor[i][j]) > 0.7) and (i != j):
            c.append(i)
print("Columns with >70%")
print(np.unique(c))

#12. Impute missing values if present using mean of the dataset.
print(iris.isnull())
#no missing values in iris dataset

#13. Save the current dataFrame out to a new csv file.
iris.to_csv('iris_after.csv',index=False)