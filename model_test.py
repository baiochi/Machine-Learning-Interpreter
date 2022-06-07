######################################################
#                Libraries and APIs
######################################################

# Custom functions for this app
from distutils.command.build import build
from functions import *

from cmath import exp
#from msilib.schema import Error
from tkinter import Button
from typing import Union, Optional, Tuple, Any
import os

# Date handling
from datetime import datetime

# Data manipulation and math operations
import numpy as np
import pandas as pd
import re
import pickle

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# Web rendering API
import streamlit as st

# Sklearn
## Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder, label_binarize
from sklearn.compose import ColumnTransformer
## Models
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
## Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,\
	confusion_matrix, classification_report, roc_auc_score, auc, f1_score, roc_curve, plot_roc_curve, precision_recall_curve
## Handling errors
from sklearn.exceptions import NotFittedError


# Other Machine Learning estimators
import lightgbm as lgb
import xgboost as xgb
from xgboost import DMatrix

# Model Interpretation
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
import shap


######################################################
#                   Configuration
######################################################

st.set_page_config(page_title='ML Visualizer', page_icon='📈',layout='wide')

#st.session_state

if 'file_upload' not in st.session_state:
    st.session_state['file_upload'] = False
if 'data' not in st.session_state:
    st.session_state['data'] = False
if 'model' not in st.session_state:
    st.session_state['model'] = False

# get files from sample_data folder
dataset_options = sorted([file[:-4] for file in os.listdir('sample_data')])

estimator_options = ('LogisticRegression', 
                    'RandomForestClassifier', 
                    'GradientBoostingClassifier', 
                    'AdaBoostClassifier')

######################################################
#                       Main
######################################################


home_placeholder = st.empty()
show_home_page(home_placeholder)

# clear home section when data is loaded
if st.session_state['file_upload']:
    home_placeholder.empty()

# Sidebar Settings
st.sidebar.header('Settings')
## Select dataset
with st.sidebar.expander('Select a dataset'):
    choice = st.radio('Options:', options=('Sample data','Upload file'))

    # Sample dataset choice
    if choice == 'Sample data':
        # Select a dataset
        sample_data = st.selectbox('Select a sample dataframe:', options=dataset_options)
        # Read data and store information on session state
        st.session_state['data'] = read_sample_data(sample_data)

    # Upload a file choice
    elif choice == 'Upload file':
        st.session_state['file_upload'] = st.file_uploader(label='Or upload a csv file.', type='csv')
        # Run if file is uploaded
        if st.session_state['file_upload']:
            # Return dataframe and a list to choose target/id columns
            df, column_selector = read_upload_file(st.session_state['file_upload'])
            # Store dataframe information on session state
            st.session_state['data']['df'] = df

## Options to show dataframe preview
if st.sidebar.checkbox('Dataframe preview'):
    st.subheader('Dataframe preview')
    st.dataframe(st.session_state['data']['df'], height=195)
    home_placeholder.empty()

# Settings for Uploaded File
if st.session_state['file_upload']:

    # target and features settings
    settings = target_features_settings(column_selector)
    st.session_state['data'].update(settings)

    # split data into train/test
    settings = test_train_split()
    st.session_state['data'].update(settings)

    # numeric data settings
    settings = numerical_transformer()
    st.session_state['data'].update(settings)

    # categorical data settings
    settings = categorical_transformer()
    st.session_state['data'].update(settings)

    # info summary
    options_summary(**st.session_state['data'])


# Select estimator
estimator = st.sidebar.selectbox('Select your model', options=estimator_options)
# Open sidebar with estimator params
model_params = configure_estimator_params(eval(estimator))

# Create model
# For uploaed file
if st.session_state['file_upload']:
    st.session_state['data'] = build_pipeline(estimator=eval(estimator), **st.session_state['data'])
    model = st.session_state['data']['pipeline']
# For sample data
else:
    model = eval(estimator)(**model_params, random_state=42)

# Button to fit model
with st.sidebar.form(key='run_model'):
    submitted = st.form_submit_button('Run model')
    if submitted:
        # Run model
        st.session_state['model'] = fit_model(model,
                                                X=st.session_state['data']['X_train'], 
                                                y=st.session_state['data']['y_train'])
        # Clean homepage after model is fitted
        home_placeholder.empty()
          
# Display metrics
if st.session_state['model']:
    
    try:
        st.subheader(f'{estimator} Metrics')
        display_metrics(model, **st.session_state['data'])

    except NotFittedError:
        not_fitted_error()
    # except (ValueError, TypeError, AttributeError) as error:
    #     collapsed_expander_bug()



#st.session_state

#st.button('Whatever', help='Visit [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html) for parameter details')

