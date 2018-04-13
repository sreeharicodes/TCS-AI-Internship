import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1. Read and merge the dataset into single dataframe
df1 =pd.read_csv('ign.csv')
df2 = pd.read_csv('ign_score.csv')
df = pd.merge(df1,df2, on='id')

df.iloc[18365,11] = '7.0'
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]

#2. Provide the names of 10 movies rated highest.
top = df.sort_values(by='score',ascending=False)
print(top[:10]['title'])

#3. Rank the movie names by their highest average rating scores.
df['score']=df['score'].astype(float)
rank = df.groupby('title').score.mean().sort_values(ascending=False)
print(rank)

#4. Plot movie scores across each genre.
df.hist(by="genre", column="score", figsize=(25,50),grid =True)
plt.show()

#5. Find the group that provides the highest average movie ratings when split into genre groups.
hamr = df.groupby('genre')[['score']].mean()
hamr = hamr.sort_values(by='score',ascending=False)
print(hamr.head(1))

#6. Provide a table with the average rating of a movie by each genre group along with the movie title.
table = pd.pivot_table(df, values='score', index=['genre', 'title'])
print(table)