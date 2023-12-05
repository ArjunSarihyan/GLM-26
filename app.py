from flask import Flask, request, jsonify
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

import GLM_Model_26 as glm_model

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the GLM Model Execution App!"

@app.route('/run_model', methods=['POST'])
def run_model():
    if request.method == 'POST':
        data = request.get_json()
        train_file = data.get('train_file')
        test_file = data.get('test_file')

        # Execute the GLM Model script with provided train and test files
        try:
            raw_train, raw_test, target_counter = glm_model.load_libraries_and_data(train_file, test_file)
            train_imputed_std, val, imputer, std_scaler, test = glm_model.feature_engineering(raw_train)
            glm_model.visualize_correlation(train_imputed_std)
            var_reduced = glm_model.perform_feature_selection(train_imputed_std)
            logistic_summary, result, variables = glm_model.prepare_train_set(train_imputed_std, var_reduced)
            val_imputed_std = glm_model.prepare_validation_set(val, imputer, std_scaler)
            test_imputed_std = glm_model.prepare_test_set(test, imputer, std_scaler)
            train_c_stat, val_c_stat, test_c_stat, grouped_bins = glm_model.outcomes_train(result, train_imputed_std, val_imputed_std, test_imputed_std, variables)
            final_model_summary, train_c_stat, grouped_bins = glm_model.finalize_model(train_imputed_std, val_imputed_std, test_imputed_std, var_reduced)    
    
            return jsonify({'message': 'Model execution completed successfully!'})
        
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'message': 'Invalid request method!'})

if __name__ == '__main__':
    app.run(port=1313)
