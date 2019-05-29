'''
The following functions were adapted from Rayid Ghani's 
repository and adapted with permission form the author. 
The link to the source is:
https://github.com/rayidghani/magicloops

'''

from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#from sklearn.grid_search import ParameterGrid
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

import pipeline as p_l


#Features we want to turn into quantiles
TO_QUANTILE = ['total_price_including_optional_support']
#Features we want to turn into dummies for their categorical nature
TO_DUMMIES = ['school_metro', 
               'primary_focus_area', 'resource_type', 'poverty_level',
              'grade_level']


#Features we want to drop, mainly the ids, latitude and longitude

DROPS = ['projectid', 'teacher_acctid', 'schoolid',
            'school_ncesid', 'school_latitude', 
            'school_longitude', 'school_district', 
            'school_county', 'secondary_focus_subject',
            'secondary_focus_area','primary_focus_subject','school_state','school_city']

#Features we already used
UNNECESARY = ['datefullyfunded', 'days_financed']

#Features we want to impute
TO_FILL = ['primary_focus_area',
               'resource_type', 'grade_level', 'students_reached']


def clean_data(data,quantile):
    '''
    Cleans the data with the following actions:
    - Drops features 
    - Fills missing values
    - Turns features into quantiles
    - Turns features into dummies
    '''
    p_l.drop_features(data, DROPS)
    p_l.drop_features(data, UNNECESARY)
    p_l.fill_missing(data,TO_FILL,form = False)
    p_l.features_quantile(data,TO_QUANTILE,quantile)
    p_l.drop_features(data, TO_QUANTILE)
    data = p_l.to_dummies(data,TO_DUMMIES)
    p_l.drop_features(data, TO_DUMMIES)
    return data


def temp_val(start_time, end_time, window_train, window_test,less_days):
    '''
    Creates a list that especifies how the temporal windows should be
    Inputs:
        start_time(str): start time of dataset
        end_time(str): end time of dataset
        window_train(int): months for amplifyhing training dataset
        window_test(int): months for amplifyhing testing dataset
    Returns: 
        list of validated dates
    '''
    validation_lst = []
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
    test_end_time = end_time_date - relativedelta(days=+1)
    train_start_time = start_time_date

    while test_end_time <= end_time_date:
        train_end_time = train_start_time + relativedelta(months=+window_train) - relativedelta(days=+less_days)
        test_start_time = train_end_time + relativedelta(days=+less_days) + relativedelta(days=+1)
        test_end_time = test_start_time + relativedelta(months=+window_test)
        validation_lst.append([train_start_time,train_end_time,test_start_time,test_end_time])
        window_train +=6
    return validation_lst
    
def temp_spl(data,temp_var,validation_elem, label):

    '''
    creates training and testing sets based on temporal validtion lists
    Inputs:
        data(DataFrame): data we want to work with
        temp_va(str): feature that indicates time
        validation_elem(list): list of training start time, end time, testing start time, end time
        label(str): label name 
    Returns: 
         training and testing sets
    '''

    train_start,train_end,test_start,test_end = validation_elem
    data[temp_var] = pd.to_datetime(data[temp_var])
    train_data = data[(train_start <= data[temp_var]) & ( data[temp_var] <= train_end)]
    X_train = train_data.drop([label,temp_var],1)
    y_train = np.array(train_data[label])
    test_data = data[(test_start <= data[temp_var] ) & (data[temp_var]<= test_end)]
    X_test = test_data.drop([label,temp_var],1)
    y_test = np.array(test_data[label])
    return X_train, X_test, y_train, y_test 


# for jupyter notebooks
#%matplotlib inline

# if you're running this in a jupyter notebook, print out the graphs
NOTEBOOK = 0

def define_clfs_params(grid_size):
    '''
    Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    Inputs:
        grid_size(str): Can be either 'large_grid', 'small_grid' or 'test_grid'
    Returns:
        clfs(ditionary): classifiers that will be used
        grid size(dictionary): parameters that will be used 
    '''

    ## dictionary of classifier methods. RF: Random Forests, ET: Extra Trees Classifier, 
    ## AB: Boosting Classifier (ADA), GB: Gradient Boosting Classifier, LR: Logistic Regression, 
    ## SVM: Support Vector Machines, DT: Decision Trees, KNN: K-Nearest Neighbors, 
    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3), 
        'BAG': BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, n_estimators = 20) 
            }

    
    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators' : [5,10, 20], 'max_samples' : [.25, .5, .75]}
       }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.1,0.5],'subsample' : [0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.1],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators' : [5,10], 'max_samples' : [.25, .5] } 
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'BAG': {'n_estimators' : [5], 'max_samples' : [.25] } 

           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

def roc_auc_sc(y_test_sorted,y_pred_probs_sorted):
    '''
    Gives the auc score 
    Input:
        y_test_sorted(array): true label from testing data
        y_pred_probs_sorted: predicted probabilities for label
    Returns:
        auc score
    '''
    return metrics.roc_auc_score(y_test_sorted,y_pred_probs_sorted)


def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays

    '''
    Order the specified arrays into descending (see which ones got the highest score)

    Intputs: 
        l1: first array
        l2: second array
    Returns: 
        The sorted data
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    '''
    Makes the cutoff for keeping only the observations that are within the top k%
    Input:
        y_scores: Predicted label
        k: threshold for keeping the points
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def f1_at_k(y_true, y_scores, k):
    '''
    Calculates the F1 score of a determined k% testing sample

    Inputs:
        y_true: label from test data
        y_scores: predicted label for test data
        k: k%, the % value we want to predict

    Returns: 
        F1 score  
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    precision = precision_score(y_true_sorted, preds_at_k)
    prec_rec = precision + recall
    if prec_rec == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
    return F1

def clf_loop(models_to_run, clfs, grid, data, temp_var, label, validation_lst, metric_lst, k_lst,quantiles, rococo = True):
    '''
    Runs the loop using models_to_run, clfs, gridm and the data
    Inputs:
        models_to_run(list): list of models to run
        clfs(dictionary of objects): dictionary with model objects
        grid(str): parameter options
        temp_var(str): temporal feature
        label(str): label feature
        validation_lst(list): list of dates for spliting data temporally
    '''


    ### Creates the final dataframe with the needed columns
    colnames = ['train_end_date','model_type','clf', 'parameters', 'str_param'] 
    colnames = colnames + define_colnames(metric_lst,k_lst)

    results_df =  pd.DataFrame(columns=(colnames))

    for elem in validation_lst:
        # create training and valdation sets
        X_train, X_test, y_train, y_test = temp_spl(data,temp_var,elem,label)       
        X_train = clean_data(X_train,quantiles)
        X_test = clean_data(X_test,quantiles)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    #y_pred = clf.fit(X_train, y_train).predict(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    #y_pred_probs_sorted, y_pred_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_pred, y_test), reverse=True))
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs,y_test), reverse=True))
                    #y_pred_sorted, y_test_sorted_ = zip(*sorted(zip(y_pred,y_test), reverse=True))
                    
                    final_lst = evaluate_models(metric_lst, k_lst,y_test_sorted,y_pred_probs_sorted, rococo)
                    total_lst = [elem[1],models_to_run[index],clf, p, str(p)] + final_lst
                    results_df.loc[len(results_df)] = total_lst

                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df


def evaluate_models(metric_lst, k_lst,y_test_sorted,y_pred_probs_sorted, rococo = True):
    '''
    Creates a list of the evaluation metrics especified. 

    Inputs: 
        metric_lst (list)
        precision_lst(list)
        y_test_sorted(array)
        y_pred_probs(array)

    '''
    final_lst = []
    mtr_lst = metric_lst
    if rococo == True:
        final_lst.append(roc_auc_sc(y_test_sorted,y_pred_probs_sorted))
    for metr in mtr_lst:
        for prec in k_lst:
            final_lst.append(metr(y_test_sorted,y_pred_probs_sorted,prec))

    return final_lst


def define_colnames(metric_lst,k_lst, rococo = True):
    '''
    Defines the names we want to assign to the columns
    '''

    final_lst = []
    names = {roc_auc_sc:'roc_auc',precision_at_k:'p_at_',recall_at_k: 'r_at_',f1_at_k: 'f1_at_'}
    mtr_lst = metric_lst
    if rococo == True:
        final_lst.append(names[roc_auc_sc])
    print(final_lst)
    for mtr in mtr_lst:
        for k in k_lst:
            final_lst.append(names[mtr] +'{}'.format(k))
    return final_lst


def rank_classifiers(df,feat,criteria):
    aa = df.groupby([feat])
    lst_to_keep = []
    output_df = None
    for key, gr in aa:
        add_ord = gr[[criteria]]
        add_ord.drop_duplicates(inplace = True)
        add_ord.sort_values(by =criteria,ascending=False,inplace=True)
        add_ord[str(key)] = np.arange(len(add_ord))
        gr.sort_values(by=criteria, ascending=False,inplace=True)
        total_df = pd.merge(gr,add_ord,on = criteria)
        total_df = total_df[['model_type','clf','parameters','str_param',str(key)]]
        if output_df is None:
            output_df = total_df
        else:
            output_df = pd.merge(output_df,total_df,on = ['model_type', 'str_param'])
        lst_to_keep.append(str(key))
    lst_to_keep = ['model_type','clf','parameters','str_param'] + lst_to_keep 
        
    return output_df[lst_to_keep]

def accurate(y_test_sorted_,y_pred_sorted):
    '''
    Calculates the accuracy score of a determined k% testing sample

    Inputs:
        y_test_sorted: label from test data
        y_pred_sorted: predicted label for test data

    Returns: 
        accuracy score  
    '''

    accure = metrics.accuracy_score(y_test_sorted_,y_pred_sorted)
    return accure

def plot_precision_recall_n(y_true, y_prob, model_name):

    '''
    Plots the precision-recall sample for the specified model

    Inputs:
        y_test_sorted_: label from test data
        y_pred_sorted: predicted label for test data

    Returns: 
        Image with the precision recall curve
    '''

    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()




def plot_roc(name, probs, true, output_type):
    '''
    Plots the AUC for the specified model

    Input:
        name(string): name we want to assign to the model
        probs(array): predicted label for test data
        true(array): label for true data
        output_types()
    
    '''
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()


