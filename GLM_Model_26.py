#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
#loading visualization library
import bokeh
from collections import Counter
import collections as ccc
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

def load_libraries_and_data(train_file, test_file):
    """
    Load necessary libraries and data files.
    Describe the target variable and display data type information.
    Investigate object columns and print unique values.
    """
    print("Reading libraries and data...")
    print("python version " + sys.version)
    print('numpy version ' + np.__version__)
    print('pandas version ' + pd.__version__)
    print('sklern version ' + '0.23.1')
    print('bokeh version ' + bokeh.__version__)
    print('statsmodels version ' + '0.9.0')
    
    raw_train = pd.read_csv(train_file)
    raw_test = pd.read_csv(test_file)
    
    print("\nDescribing the target variable...")
    target_counter = Counter(raw_train.y)
    
    print("Overview of data types:")
    print("object dtype:", raw_train.columns[raw_train.dtypes == 'object'].tolist())
    print("int64 dtype:", raw_train.columns[raw_train.dtypes == 'int'].tolist())
    print("The rest of the columns have float64 dtypes.")
    
    print("\nInvestigating object columns...\n")
    col_obj = raw_train.columns[raw_train.dtypes == 'object']
    for i in range(len(col_obj)):
        if len(raw_train[col_obj[i]].unique()) > 13:
            print(col_obj[i]+":", "Unique Values:", np.append(raw_train[col_obj[i]].unique()[:13], "..."))
        else:
            print(col_obj[i]+":", "Unique Values:", raw_train[col_obj[i]].unique())
    
    del col_obj
    
    return raw_train, raw_test, target_counter


# In[3]:

def feature_engineering(raw_train):
    train_val = raw_train.copy(deep=True)

    # Fixing the money and percents
    train_val['x12'] = train_val['x12'].str.replace('$', '')
    train_val['x12'] = train_val['x12'].str.replace(',', '')
    train_val['x12'] = train_val['x12'].str.replace(')', '')
    train_val['x12'] = train_val['x12'].str.replace('(', '-')
    train_val['x12'] = train_val['x12'].astype(float)
    train_val['x63'] = train_val['x63'].str.replace('%', '')
    train_val['x63'] = train_val['x63'].astype(float)

    # Creating the train/val/test set
    x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

    # Smashing sets back together
    train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
    val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
    test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)


    # With mean imputation from Train set
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    train_imputed = pd.DataFrame(imputer.fit_transform(train.drop(columns=['y', 'x5', 'x31',  'x81' ,'x82'])), columns=train.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
    std_scaler = StandardScaler()
    train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

    # Create dummies
    dumb5 = pd.get_dummies(train['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb5], axis=1, sort=False)
    
    dumb31 = pd.get_dummies(train['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb31], axis=1, sort=False)
    dumb81 = pd.get_dummies(train['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb81], axis=1, sort=False)
    
    dumb82 = pd.get_dummies(train['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb82], axis=1, sort=False)
    
    train_imputed_std = pd.concat([train_imputed_std, train['y']], axis=1, sort=False)


    del dumb5, dumb31, dumb81, dumb82

    # Print imputer statistics and variance
    print("\n Imputer statistics:\n",imputer.statistics_)
    print("\n Train imputed variance:\n",train_imputed.var())
    
    return train_imputed_std, val, imputer, std_scaler, test


# In[4]:

def visualize_correlation(df):
    """
    This function visualizes the correlations in a heatmap.
    It helps in exploratory analysis to identify high pairwise correlations.
    If there are few variables correlated with the target, it suggests using an L2 penalty.
    If there are many variables correlated with y, it recommends using an L2 penalty.
    """
    sns.set(style='white')
    corr = df.corr()

    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1)
    sns.heatmap(data=corr,
                center=0,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, linewidth=0.5)
    plt.show()


# In[5]:

def perform_feature_selection(train_imputed_std):
    """
    Perform initial feature selection using Logistic Regression with L1 penalty.

    Parameters:
    train_imputed_std (DataFrame): Data for feature selection.

    Returns:
    DataFrame: Top 25 features selected based on coefficients squared.
    """
    exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
    exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
    exploratory_results['coefs'] = exploratory_LR.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
    var_reduced = exploratory_results.nlargest(25,'coefs_squared')
    return var_reduced


# In[6]:

def prepare_train_set(train_imputed_std, var_reduced):
    variables = var_reduced['name'].to_list()
    logit = sm.Logit(train_imputed_std['y'], train_imputed_std[variables])
    result = logit.fit()
    summary = result.summary()
    print(summary)
    return summary, result, variables


# In[7]:

def prepare_validation_set(val, imputer, std_scaler):
    val_imputed = pd.DataFrame(imputer.transform(val.drop(columns=['y', 'x5', 'x31', 'x81', 'x82'])), columns=val.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
    val_imputed_std = pd.DataFrame(std_scaler.transform(val_imputed), columns=val_imputed.columns)
    
    for col in ['x5', 'x31', 'x81', 'x82']:
        dummies = pd.get_dummies(val[col], drop_first=True, prefix=col, prefix_sep='_', dummy_na=True)
        val_imputed_std = pd.concat([val_imputed_std, dummies], axis=1, sort=False)
    
    val_imputed_std = pd.concat([val_imputed_std, val['y']], axis=1, sort=False)
    print("\n val_imputed_std.head",val_imputed_std.head())
    print("\n val_imputed_std.columns",val_imputed_std.columns)
    
    return val_imputed_std


# In[8]:

def prepare_test_set(test, imputer, std_scaler):
    test_imputed = pd.DataFrame(imputer.transform(test.drop(columns=['y', 'x5', 'x31', 'x81', 'x82'])), columns=test.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
    test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=test_imputed.columns)
    
    for col in ['x5', 'x31', 'x81', 'x82']:
        dummies = pd.get_dummies(test[col], drop_first=True, prefix=col, prefix_sep='_', dummy_na=True)
        test_imputed_std = pd.concat([test_imputed_std, dummies], axis=1, sort=False)
    
    test_imputed_std = pd.concat([test_imputed_std, test['y']], axis=1, sort=False)
    print("\n test_imputed_std.head()",test_imputed_std.head())
    print("\n test_imputed_std.columns",test_imputed_std.columns)
    
    return test_imputed_std


# In[9]:

def outcomes_train(result, train_imputed_std, val_imputed_std, test_imputed_std, variables):
    """
    Calculate outcomes and C-statistics for training, validation, and test sets.

    Parameters:
    result (statsmodels.discrete.discrete_model.BinaryResultsWrapper): Fitted logistic regression model.
    train_imputed_std (DataFrame): Imputed and standardized training data.
    val_imputed_std (DataFrame): Imputed and standardized validation data.
    test_imputed_std (DataFrame): Imputed and standardized test data.
    variables (list): List of variables used in the logistic regression model.

    Returns:
    tuple: C-statistics for train, validation, and test sets.
    """
    Outcomes_train = pd.DataFrame(result.predict(train_imputed_std[variables])).rename(columns={0:'probs'})
    Outcomes_train['y'] = train_imputed_std['y']
    print('\n The C-Statistics for Training set is ', roc_auc_score(Outcomes_train['y'], Outcomes_train['probs']))
    
    Outcomes_val = pd.DataFrame(result.predict(val_imputed_std[variables])).rename(columns={0:'probs'})
    Outcomes_val['y'] = val_imputed_std['y']
    print('\n The C-Statistics for Validation set is ', roc_auc_score(Outcomes_val['y'], Outcomes_val['probs']))
    
    Outcomes_test = pd.DataFrame(result.predict(test_imputed_std[variables])).rename(columns={0:'probs'})
    Outcomes_test['y'] = test_imputed_std['y']
    print('\n The C-Statistics for Test set is ', roc_auc_score(Outcomes_test['y'], Outcomes_test['probs']))
    
    Outcomes_train['prob_bin'] = pd.qcut(Outcomes_train['probs'], q=20)
    grouped_bins = Outcomes_train.groupby(['prob_bin'])['y'].sum()
    
    print('\n Grouped Bins:', grouped_bins)
    
    return (
        roc_auc_score(Outcomes_train['y'], Outcomes_train['probs']),
        roc_auc_score(Outcomes_val['y'], Outcomes_val['probs']),
        roc_auc_score(Outcomes_test['y'], Outcomes_test['probs']),
        grouped_bins
    )


# In[10]:

def finalize_model(train_imputed_std, val_imputed_std, test_imputed_std, var_reduced):
    """
    Finalize the model by refitting on all training data and evaluating its performance.

    Parameters:
    train_imputed_std (DataFrame): Imputed and standardized training data.
    val_imputed_std (DataFrame): Imputed and standardized validation data.
    test_imputed_std (DataFrame): Imputed and standardized test data.
    var_reduced (DataFrame): DataFrame containing reduced variables from feature selection.

    Returns:
    dict: Dictionary containing final model summary and evaluation metrics.
    """
    train_and_val = pd.concat([train_imputed_std, val_imputed_std])
    all_train = pd.concat([train_and_val, test_imputed_std])
    variables = var_reduced['name'].to_list()
    
    final_logit = sm.Logit(all_train['y'], all_train[variables])
    final_result = final_logit.fit()
    
    Outcomes_train_final = pd.DataFrame(final_result.predict(all_train[variables])).rename(columns={0:'probs'})
    Outcomes_train_final['y'] = all_train['y']
    train_c_stat = roc_auc_score(Outcomes_train_final['y'], Outcomes_train_final['probs'])
    
    grouped_bins = Outcomes_train_final.groupby(pd.qcut(Outcomes_train_final['probs'], q=20))['y'].sum()
    
    print("\nFinal Result Summary:\n", final_result.summary())
    print('\nThe C-Statistics is', train_c_stat)
    
    Outcomes_train_final['prob_bin'] = pd.qcut(Outcomes_train_final['probs'], q=20)
    grouped_bins = Outcomes_train_final.groupby(['prob_bin'])['y'].sum()
    print("\nGrouped Bins:", grouped_bins)
    
    return final_result.summary(), train_c_stat, grouped_bins


# In[11]:

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python GLM_Model_26.py <train_file.csv> <test_file.csv>")
        sys.exit(1)
        
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    # Load libraries and data
    raw_train, raw_test, target_counter = load_libraries_and_data(train_file, test_file)

    # Feature engineering
    train_imputed_std, val, imputer, std_scaler, test = feature_engineering(raw_train)

    # Visualize correlation
    visualize_correlation(train_imputed_std)
    
    # Perform feature selection
    var_reduced = perform_feature_selection(train_imputed_std)
    
    # Prepare train set
    logistic_summary, result, variables = prepare_train_set(train_imputed_std, var_reduced)
    #print(logistic_summary)
    
    # Prepare validation and test sets
    val_imputed_std = prepare_validation_set(val, imputer, std_scaler)
    
    test_imputed_std = prepare_test_set(test, imputer, std_scaler)
    
    train_c_stat, val_c_stat, test_c_stat, grouped_bins = outcomes_train(result, train_imputed_std, val_imputed_std, test_imputed_std, variables)
    #print('\n Grouped Bins:', grouped_bins)
    
    # Finalize the model
    final_model_summary, train_c_stat, grouped_bins = finalize_model(train_imputed_std, val_imputed_std, test_imputed_std, var_reduced)    
    