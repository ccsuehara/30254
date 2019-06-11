import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn import metrics


SEED = 1234



#Step 1: Read the data
def read_data(filename):
    '''
    Read the data and convert it to a dataframe

    Input:
        filename(csv): data file
    Output: 
        dataframe

    '''
    data =pd.read_csv(filename)
    return data

#Step 2: Explore


def df_shape(data):
    '''
    Returns the shape of dataframe
    Input:
        Dataframe 
    '''
    return data.shape

def df_columns(data):
    '''
    Returns the column names of dataframe
    Input:
        Dataframe 
    '''
    return data.columns

def df_head(data):
    '''
    Returns the first 5 rows of dataframe
    Input:
        Dataframe 
    '''
    return data.head()

def df_info(data):
    '''
    Returns information dataframe
    Input:
        Dataframe 
    '''
    return data.info()

def df_description(data):
    '''
    Returns the description of dataframe
    Input:
        Dataframe 
    '''
    return data.describe()

def df_missing_values(data):
    '''
    Returns the description dataframe
    Input:
        Dataframe 
    '''
    return data.isna().sum()

def drop_features(data, features_lst):
    '''
    Drops list of features specified
    Input:
        data(dataframe): Dataframe 
        features_lst: Features to eliminate
    Returns:
        Nothing, it just modifies the dataframe
    '''
    data.drop(features_lst,axis =1, inplace = True)



def histogram_by_group(data,label):
    '''
    Plots histogram of all features in dataframe differentiated
    by all categories of the label 
    Input:
        data(dataframe)
        label(str): label name
    Output:
        Histogram plots
    '''
    for i, col in enumerate(data.columns):
        plt.figure(i)
        data_gr = data.groupby(label)[col]
        data_gr.plot(kind='hist', figsize=[12,6], 
                     alpha=.4, title = col, legend=True)


def correlation_matrix(data):
    '''
    Plots heatmap with  information about correlation 
    between all pairs of fetures + label
    Input:
        data(dataframe)       
    '''
    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True, annot = True);


def missing_val_cols(data):
    '''
    Check whick features have missing values
    Input:
        data(dataframe)
    Output:
        list of features with missing values 
    '''
    missing_lst = []
    for _, col in enumerate(data.columns):
        a = data[col].isna().sum()
        if a !=0:
            missing_lst.append(col)
    return missing_lst
        

#Fill the variables with the median
def fill_missing(data, missing_lst,form = True):
    '''
    Fills features missing values with mean or median
    Input:
        data(dataframe)
        missing_lst(list):list of features with missing values
        form: True if filled with mean, False if filled with median
    '''
    for col in missing_lst:
        if data[col].dtype == np.object:
            data[col] = data[col].fillna(data[col].mode().iloc[0])
        else:
            if form:
                data[col].fillna(data[col].mean(), 
                                 inplace = True)
            else: 
                data[col].fillna(data[col].median(), 
                                 inplace = True)



def features_quantile(data, feature_lst,q):
    '''
    Discretize a variable according to quantiles
    Input: 
        data(dataframe)
        feature_lst: list of features to discretize
        q: number of divisions (quantiles)
    '''
    for _, col in enumerate(feature_lst):
        col_q = col + '_q{}'.format(q)
        data[col_q] = pd.qcut(data[col],q, labels =False)
        
def to_dummies(data,feature_lst):
    '''
    Turns a categoric variable into dummies
    Input:
        data(dataframe)
        feature_lst(list):list of features to turn
        into dummy
    Returns: 
        Dataframe with new columns for dummies

    '''
    for feat in feature_lst:
        df_feat = pd.get_dummies(data[feat], prefix = feat)
        data = data.join(df_feat)
    return data



def select_and_split_data(data,label,t_size,seed):
    '''
    Divides data into training and testing sets
    Input:
        data(dataframe)
        label(str): label columns
        t_size(float): training size
        seed (int): seed
    '''

    X = np.array(data.drop([label],1))
    y = np.array(data[label])
    X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y, 
                                      test_size = t_size,
                                      random_state = seed)
    return X_train, X_test, y_train, y_test


def build_knn_classifiers(X_train,y_train,X_test, num_neighbors, weights):
    '''
    Builds K-Nearest Neighbor models 
    Input:
        X_train(numpy array): training set for features
        y_train(numpy array): training set for labels
        num_neighbors(list of integers):list of possible number of neighbors
        weights(list of strings): list of types of weights
    Returns:
        list of models
    '''
    model_lst = []
    for k in num_neighbors:
        for w in weights:
            knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights=w)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            model_lst.append([k,w,knn,y_pred])
    return model_lst


def evaluate_model(model_lst,y_test):
    '''
    Evaluates classifier according to criterias of:
    sensitivity, specificity, false positive rate, 
    precision and accuracy.

    Input:

    model_lst: list of models
    y_test (numpy array): testing set for labels

    '''
    print("# Neighbors | Weights | Sentitivity | Specificity | False Positive Rate | Precision | Accuracy ")
    for model in model_lst:
        confusion = metrics.confusion_matrix(y_test, model[3])
        tp = confusion[1][1]
        tn = confusion[0][0]
        fp = confusion[0][1]
        fn = confusion[1][0]
        sensitivity = tp/(tp + fn)
        specificity = tn/(tp + fn)
        fp_rate = fp/(tn + fp)
        precision = tp/(tp + fp)
        accuracy = metrics.accuracy_score(y_test, model[3])
        print("{} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} ".format(model[0],model[1], sensitivity, specificity, fp_rate, 
              precision, accuracy))