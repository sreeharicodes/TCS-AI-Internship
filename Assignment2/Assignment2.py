import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1. Read and merge the dataset into single dataframe
df1 =pd.read_csv('ign.csv')
df2 = pd.read_csv('ign_score.csv')
df = pd.merge(df1,df2, on='id')

df.iloc[18365,11] = '6.4'
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
#2. Provide the names of 10 movies rated highest.
top = df.sort_values(by='score',ascending=False)
top = top['title'].unique()[:10]
print(top)

#3. Rank the movie names by their highest average rating scores.
#found that score is not numeric ,so while converting to float one row of score \
# field is playstaion beta so i gave an average value 7 instead of that converted the score field to float
# print("df.iloc[18365,11] >before")
# print(df.iloc[18365,11])
# df.iloc[18365,11] = '7.0'
# print("df.iloc[18365,11] >after")
# print(df.iloc[18365,11])

df['score']=df['score'].astype(float)

rank = df.groupby('title').score.mean().sort_values(ascending=False)
r = rank.rank(method ='dense',ascending=False)
#print(rank)
df = df.assign(Rank=r)
print(df)