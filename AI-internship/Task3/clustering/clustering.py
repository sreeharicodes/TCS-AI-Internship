import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def readdata():
    try:
        df = pd.read_csv('Wholesale customers data.csv')
        return df
    except:
        print("data file not found")
    
def analysis(df):
    sns.pairplot(data = df)
    plt.show()

def feature_importance(df):
    features = list(df.columns)
    for feature in features:
        new_data = df.drop([feature], axis=1)
        new_feature = pd.DataFrame(df.loc[:, feature])
        X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.25, random_state=42)
        dtr = DecisionTreeRegressor(random_state=42)
        dtr.fit(X_train, y_train)
        score = dtr.score(X_test, y_test)
        print('R2 score for {} as dependent variable: {}'.format(feature, score))
def feature_scaling(df):
    log_data = np.log(df)
    #analysis(log_data)
    #Outlier Detection
    outliers_lst = []
    for feature in log_data.columns:
        Q1 = np.percentile(log_data.loc[:, feature], 25)
        Q3 = np.percentile(log_data.loc[:, feature], 75)
        step = 1.5 * (Q3 - Q1)
        print("Data points considered outliers for the feature '{}':".format(feature))
        outliers_rows = log_data.loc[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step)), :]
        outliers_lst.append(list(outliers_rows.index))
    outliers = list(itertools.chain.from_iterable(outliers_lst))
    uniq_outliers = list(set(outliers))
    dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
    print('Outliers list:\n', uniq_outliers)
    print('Length of outliers list:\n', len(uniq_outliers))
    print('Duplicate list:\n', dup_outliers)
    print('Length of duplicates list:\n', len(dup_outliers))
    # Remove duplicate outliers
    # Only 5 specified
    good_data = log_data.drop(log_data.index[dup_outliers]).reset_index(drop=True)
    # Original Data
    print('Original shape of data:\n', df.shape)
    # Processed Data
    print('New shape of data:\n', good_data.shape)
    return good_data
def feature_transformation(df):
    pca = PCA(n_components=6)
    pca.fit(df)

def dimension_reduction(df):
    pca = PCA(n_components=2)
    pca.fit(df)
    reduced_data = pca.transform(df)
    reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
    return reduced_data
def knn(df):
    range_n_clusters = list(range(2, 11))
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters).fit(df)
        preds = clusterer.predict(df)
        score = silhouette_score(df, preds, metric='euclidean')
        print("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))
    clusterer = KMeans(n_clusters=2).fit(df)
    preds = clusterer.predict(df)
    centers = clusterer.cluster_centers_
    print(centers)



def main():
    df = readdata()
    df.drop(['Channel','Region'], inplace =  True, axis = 1)
    analysis(df)
    feature_importance(df)
    good_data = feature_scaling(df)
    feature_transformation(good_data)
    red_data = dimension_reduction(good_data)
    knn(red_data)
    #no of cluster 2 beacause silhouette score is more when no of cluster = 2
    

if __name__ == '__main__':
    main()
