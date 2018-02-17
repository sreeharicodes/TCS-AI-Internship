import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df1 =pd.read_csv('ign.csv')
df2 = pd.read_csv('ign_score.csv')
#1. Read and merge the dataset into single dataframe
df = pd.merge(df1,df2, on='id')

#2. Provide the names of 10 movies rated highest.
top = df.sort_values(by='score',ascending=False)[:10]
print(top['title'])