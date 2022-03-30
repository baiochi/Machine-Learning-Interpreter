######################################################
#                Libraries and APIs
######################################################

# Custom functions for this app
from functions import *

from cmath import exp
#from msilib.schema import Error
from tkinter import Button
from typing import Union, Optional, Tuple, Any

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

if 'file_loaded' not in st.session_state:
    st.session_state['file_loaded'] = False
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = False
if 'target_name' not in st.session_state:
    st.session_state['target_name'] = False
if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = False
if 'model' not in st.session_state:
    st.session_state['model'] = False

dataset_options = ('iris', 'penguins', 'diamonds', 'tips')

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
if st.session_state['file_loaded']:
    home_placeholder.empty()

# Sidebar Settings
st.sidebar.header('Settings')
## Select dataset
#with st.sidebar.form(key='data_selection'):
with st.sidebar.expander('Select a dataset'):
    choice = st.radio('Options:', options=('Sample data','Upload file'))
    # Sample dataset
    if choice == 'Sample data':
        st.session_state['dataset'] = st.selectbox('Select a sample dataframe:', options=dataset_options)
        st.session_state['target_name'] = None
    # Upload a file
    elif choice == 'Upload file':
        st.session_state['dataset'] = st.file_uploader(label='Or upload a csv file.', type='csv')
        if st.session_state['dataset']:
            st.session_state['dataset'] =  pd.read_csv(st.session_state['dataset'])
            st.session_state['target_name'] = st.selectbox('Select target:', options=st.session_state['dataset'].columns)
    
    # # collapse form
    # if st.form_submit_button('Load data'):
    #     st.session_state['file_loaded'] = True

## Read dataset and display informations
# if st.session_state['file_loaded']:

# Read dataframe information on session state
st.session_state['dataframe'] = read_data(st.session_state['dataset'], st.session_state['target_name'])

with st.sidebar.expander('Dataframe info'):
    st.sidebar.markdown(f"Target: `{st.session_state['target_name']}`")
# options to show df preview
if st.sidebar.checkbox('Data preview'):
    st.subheader('Data preview')
    st.dataframe(st.session_state['dataframe']['processed_df'], height=195)


## Select estimator
estimator = st.sidebar.selectbox('Select your model', options=estimator_options)

## Open sidebar with estimator params
model_params = configure_estimator_params(eval(estimator))

## Create model
model = eval(estimator)(**model_params, random_state=42)

## Run Model
if st.sidebar.button('Run model'):

    if not st.session_state['file_loaded']:
        home_placeholder.empty()
        st.error('## Please load any data before attempt to fit a model.')

    try:
        # start timerasas
        start_time =  datetime.now()
        # fit model
        model.fit(
                X=st.session_state['dataframe']['X_train'], 
                y=st.session_state['dataframe']['y_train']
        )
        # calculate total time
        end_time = datetime.now()
        total_time = f'Time to fit: ' + str(end_time - start_time).split(".")[0]
        st.session_state['model'] = model
        st.sidebar.success(total_time) 

    except NotFittedError:
        not_fitted_error()
    except (ValueError, TypeError, AttributeError) as error:
        collapsed_expander_bug()
          
# Display metrics
if st.session_state['model']:
    
    try:
        st.subheader(f'{estimator} Metrics')
        display_metrics(model, **st.session_state['dataframe'])

    except NotFittedError:
        not_fitted_error()
    # except (ValueError, TypeError, AttributeError) as error:
    #     collapsed_expander_bug()



#st.session_state

#st.button('Whatever', help='Visit [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html) for parameter details')

