import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import xgboost
import datetime 
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import warnings
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings('ignore')


def datavisualisation(df):
    sns.pairplot(df.dropna(),hue = 'click')
    plt.show()
    sns.countplot(x = 'click', data = df)
    plt.show()
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    plt.show()

def info(df):
    print(df.head())
    print(df.info())

def preprocessing(df):
    #renamebrowser
    df.loc[df.browserid.isin(['Internet Explorer', 'InternetExplorer']), 'browserid'] = 'IE'
    df.loc[df.browserid.isin(['Mozilla Firefox', 'Mozilla']), 'browserid'] = 'Firefox'
    df.loc[df.browserid=='Google Chrome', 'browserid'] = 'Chrome'
    #parse time
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def dayornight(cols):
    time = cols[0]
    if time <= 19 and time >= 6:
        return 'day'
    else:
        return 'night'

def imputation(df):
    df['siteid'].fillna(-999, inplace=True)
    df['browserid'].fillna("otherbrowser", inplace=True)
    df['devid'].fillna("otherdevice", inplace=True)
    #create meaningfull column
    df['hour'] = df['datetime'].dt.hour
    return df

def deleteunnamed(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def imbalancedcheck(train):
    Count_NC = len(train[train["click"]==0])
    Count_C = len(train[train["click"]==1])
    pnc = (Count_NC/(Count_NC+Count_C))*100
    pc = (Count_C/(Count_NC+Count_C))*100
    print("Percentage- not clicked = {}".format(pnc) )
    print("Percentage- clicked = {}".format(pc) )

def featureselection(df):
    df['d/n']=df[['hour']].apply(dayornight,axis=1)
    df = one_hot_encoding(df,'d/n')
    df = one_hot_encoding(df,'browserid')
    df = one_hot_encoding(df,'devid')
    df = one_hot_encoding(df,'countrycode')
    #print(df.info())
    #feature selection
    df = df.drop(columns=['datetime','ID','hour'])
    #print(df.info())
    X = list(df)
    X.remove('click')
    Y = df['click']
    X_ = df[X]
    X_train,X_test,y_train,y_test=train_test_split(X_,Y,test_size=0.20,random_state=5)
    #Import the random forest model.
    from sklearn.ensemble import RandomForestClassifier 
    #instantiates the model. 
    rf = RandomForestClassifier() 
    #Fit the model on  training data.
    rf.fit(X_train, y_train) 
    #And score it on  testing data.
    rf.score(X_test, y_test)
    feature_importances = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
    features = list(feature_importances.index)
    features = features[0:5]
    return features, df

def one_hot_encoding(df,label):
    one_hot=pd.get_dummies(df[label])
    del df[label]
    df=df.join(one_hot)
    return df


def undersample(normal_indices,click_indices,times, data, count_click):#times denote the normal data = times*fraud data
    Normal_indices_undersample = np.array(np.random.choice(normal_indices,(times*count_click),replace=False))
    undersample_data= np.concatenate([click_indices,Normal_indices_undersample])
    undersample_data = data.iloc[undersample_data,:]
    return(undersample_data)

def model(model,features_train,features_test,labels_train,labels_test):
    clf= model
    clf.fit(features_train,labels_train.values.ravel())
    pred=clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    cnf_matrix=confusion_matrix(labels_test,pred)
    print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    print('accuracy :', accuracy)
    print("\n")
    print('cross validationscore :', accuracy)
    fig= plt.figure(figsize=(6,3))
    print("TP",cnf_matrix[1,1,]) 
    print("TN",cnf_matrix[0,0]) 
    print("FP",cnf_matrix[0,1]) 
    print("FN",cnf_matrix[1,0]) 
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred))

def data_prepration(x):
    x_features= x.ix[:,x.columns != "click"]
    x_labels=x.ix[:,x.columns=="click"]
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.2)
    # print("length of training data")
    # print(len(x_features_train))
    # print("length of test data")
    # print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

def main():
    data = pd.read_csv('dataset/train.csv')
    datavisualisation(data)
    df = preprocessing(data)
    y = df['click']
    train, test = train_test_split(df, test_size = 0.3, stratify = y)
    train = imputation(train)
    test = imputation(test)
    train.reset_index(drop=True, inplace= True)
    test.reset_index(drop=True, inplace= True)
    train = deleteunnamed(train)
    ftrain = train
    features, ftrain = featureselection(ftrain)
    target = 'click'
    f = features
    features.append('click')
    ftrain = ftrain[features]
    imbalancedcheck(ftrain)
    data = ftrain
    click_indices= np.array(data[data.click==1].index)
    normal_indices = np.array(data[data.click==0].index)
    Count_C = len(data[data["click"]==1])
    
   
    #svm 
    print("SVM")
    for i in range(1,4):
        print("the undersample data for {} proportion".format(i))
        print()
        Undersample_data = undersample(normal_indices,click_indices,i,data,Count_C)
        print("------------------------------------------------------------")
        print()
        print("the model classification for {} proportion".format(i))
        print()
        undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=data_prepration(Undersample_data)
        data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data) 
        #the partion for whole data
        print()
        clf=SVC()
        model(clf,undersample_features_train,data_features_test,undersample_labels_train,data_labels_test)
        # here training for the undersample data but testing for whole data
        print("_________________________________________________________________________________________")
    
    
    #naive bayse
    print("naive bayse")
    for i in range(1,4):
        print("the undersample data for {} proportion".format(i))
        print()
        Undersample_data = undersample(normal_indices,click_indices,i,data,Count_C)
        print("------------------------------------------------------------")
        print()
        print("the model classification for {} proportion".format(i))
        print()
        undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=data_prepration(Undersample_data)
        data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data) 
        #the partion for whole data
        print()
        clf=GaussianNB()
        model(clf,undersample_features_train,data_features_test,undersample_labels_train,data_labels_test)
        # here training for the undersample data but testing for whole data
        print("_________________________________________________________________________________________")
    
    #xgboost
    print("Xgboost")
    for i in range(1,4):
        print("the undersample data for {} proportion".format(i))
        print()
        Undersample_data = undersample(normal_indices,click_indices,i,data,Count_C)
        print("------------------------------------------------------------")
        print()
        print("the model classification for {} proportion".format(i))
        print()
        undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=data_prepration(Undersample_data)
        data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data) 
        #the partion for whole data
        print()
        clf= xgboost.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.015)
        model(clf,undersample_features_train,data_features_test,undersample_labels_train,data_labels_test)
        # here training for the undersample data but testing for whole data
        print("_________________________________________________________________________________________")
    #after testing with 3 classifiers xgboost and svm comes prettywell
    #best classifier of propotion4
    print("svc")
    Undersample_data = undersample(normal_indices,click_indices,4,data,Count_C)
    print("------------------------------------------------------------")
    print()
    print("the model classification for 4* proportion")
    print()
    undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=data_prepration(Undersample_data)
    data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data) 
    #the partion for whole data
    print()
    clf=SVC()
    model(clf,undersample_features_train,data_features_test,undersample_labels_train,data_labels_test)
    print("xgboost")
    Undersample_data = undersample(normal_indices,click_indices,4,data,Count_C)
    print("------------------------------------------------------------")
    print()
    print("the model classification for 4* proportion")
    print()
    undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=data_prepration(Undersample_data)
    data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data) 
    #the partion for whole data
    print()
    clf=xgboost.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.015)
    model(clf,undersample_features_train,data_features_test,undersample_labels_train,data_labels_test)




    

if __name__ == '__main__':
    main()