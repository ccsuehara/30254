'''
Augmenting the pipeline for Clustering
Author: Carla Solis
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt



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


def k_means(df, k,iters):
    '''
    Calculate k means for a dataframe with all the features that are present in it

    Inputs:
        df(Dataframe): dataframe
        k (int): number of clusters we want
        iters (int): nunber of iterations we want to perform before converging to solution

    '''
    model = KMeans(n_clusters = k, max_iter = iters).fit(df)
    pred_k = model.predict(df)
    label = model.labels_
    centroids = model.cluster_centers_
    df['label'] = label



def merge_with_k(df, clusters_to_merge ,new_cluster_val):
    '''
    Merges multiple clusters into just 1 (the chosen one)
    Inputs:
        df (Dataframe): dataframe where data is
        clusters_to_merge(list of ints): list of clusters we want to merge
        new_cluster_val(int): cluster that will last
    '''

    mergings = df.replace(to_replace = clusters_to_merge, 
                          value = new_cluster_val)
    return mergings




def recluster_new_k(df, new_k,label_name,iters):
    '''
    Reclusters a dataframe already clustered
    Inputs: 
        df (Dataframe): dataframe where data is
        new_k (int): number of clusters we want NOW
        label_name(string): name of string that will be removed
        iters(int): nunber of iterations we want to perform before converging to solution


    '''
    if label_name in df.columns:
        new_k_df = df.drop(label_name, axis=1)
        k_means(new_k_df,new_k,iters)
    else:
        raise Exception('no cluster was previously made!')

    return new_k_df


def split_to_many(df, splitter_k, new_sub_k,label):
    '''
    

    '''
    first_chunk = df[df[label] == splitter_k]
    second_chunk = df[df[label] != splitter_k]

    new_first_chunk = re_c(first_chunk,new_sub_k, label)
    tot = new_first_chunk.append(second_chunk)
    return tot


