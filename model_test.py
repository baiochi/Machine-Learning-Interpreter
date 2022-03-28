######################################################
#                Libraries and APIs
######################################################
from typing import Union, Optional, Tuple, Any

# Date handling
from datetime import datetime

# Data manipulation and math operations
import numpy as np
import pandas as pd
import re
import pickle
import numbers

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Web rendering API
import streamlit as st

# Sklearn
#	Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
#	Models
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
#	Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,\
	ConfusionMatrixDisplay, classification_report, roc_auc_score, f1_score, roc_curve, plot_roc_curve, precision_recall_curve

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
#             Data Engineering Functions
######################################################

def get_docstring_params(estimator):
	# get full docstring
	doc_string = estimator().__doc__
	doc_string = doc_string.split('\n')
	# select only values with Parameters(start) and Attributes(end)
	r = re.compile("(.*Parameters)|(.*Attributes)")
	start_end_values = list(filter(r.match, doc_string))
	# get position of each value
	start = doc_string.index(start_end_values[0])
	end = doc_string.index(start_end_values[1])
	# new docstring
	short_docstring = '\n'.join(doc_string[start:end])
	# remove unwanted characters
	short_docstring = short_docstring.replace('`', '').replace('        -', '').replace('\t', '-')
	return short_docstring

def get_param_options(estimator):
	# empty dictionary for only parameters with multiple options
	opt_param_dict = {}

	param_section = get_docstring_params(estimator)
	# search all docstring for parameters with a list of options
	options_params = re.findall('    .* : {.*}',estimator().__doc__)
	for param in options_params:
		# divine between key and values	
		param_split = param.split(':')
		# clean key string
		param_key = param_split[0]
		param_key = param_key.replace(' ', '')
		# clean values string
		param_value = param_split[1]
		opt_param_dict[param_key] = re.findall('\'(.*?)\'', param_value)
		# some parameters are written in "" instead of ''
		if len(opt_param_dict[param_key]) == 0:
			opt_param_dict[param_key] = re.findall('\"(.*?)\"', param_value)

	return opt_param_dict

def get_default_params(estimator):
	# get section of parameters from full docstring
	param_docs = get_docstring_params(estimator)
	# get each parameter
	params = re.findall('    .* : ', param_docs)
	# clean empty characters and :
	params = [param.replace(' ','').replace(':', '') for param in params]
	# create dict with default values
	default_values = {key:value for key, value in vars(estimator()).items() if key in params}
	# get only params with multiple values
	opt_values = get_param_options(estimator)
	# update default params
	for key, value in opt_values.items():
		if default_values[key]:
			# change position of default value to first index
			value.remove(default_values[key])
			value.insert(0,default_values[key])
			# update value
			default_values[key] = value
		else: #default value == None
			value.insert(0,None)
			# update value
			default_values[key] = value

	return default_values

######################################################
#        Machine Learning Pipeline Functions
######################################################

def select_dataset(dataset):
	if dataset == 'iris':
		df = sns.load_dataset('iris')
		target_name = 'species'
	elif dataset == 'penguins':
		df = sns.load_dataset('penguins')
		target_name = 'species'
	elif dataset == 'titanic':
		df = sns.load_dataset('titanic')
		target_name = 'survived'
	return df, target_name

def prepare_data(df, target_name):
    """  
    \nPreprocess data\n----------\n
    Apply every transformation need in order to fil models, like onehot encoding and fill null values\n
    - df : `pd.DataFrame`, used to apply changes  
    - target_name : `str`, column to be predicted  
    \Returns\n----------\n
    - X :  `pd.Dataframe`, transformed features
    - y :  `pd.Series`, target values
    - target_labels : `dict`, mapping of categorical values 
    \nExample\n----------\n
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> df = sns.load_dataset('iris')
    >>> X, y, target_labels = prepare_data(df, target_name='species')
    """
    # Onehot encoding for categorical features, and fill null values
    X = pd.get_dummies(df.drop(target_name, axis=1)).fillna(0)
    # For all columns, replace non alphanumeric characters with "_"
    X.columns = [''.join(char if char.isalnum() else "_" for char in str(column)) for column in X.columns]

    # Target is from a classification problem
    if not isinstance(df[target_name], np.number):
        target = df[target_name].astype('category')
        target_labels = dict( enumerate(target.cat.categories ) )
        y = target.cat.codes
    # Target is from a regression problem
    else:
        y = df[target_name]
        target_labels = None

    return X, y, target_labels

def create_model():
    pass
######################################################
#                  Rendering Functions
######################################################

# Show estimator params in Sidebar
def configure_estimator_params(estimator):

    model_params = {}

    with st.sidebar.expander('Configure parameters'):
        for key, value in get_default_params(estimator).items():
            # True or False params
            if isinstance(value, bool):
                model_params[key] = st.checkbox(label=key, value=value)
            # float or int params
            elif isinstance(value, numbers.Number):
                model_params[key] = st.number_input(label=key, value=value)
            # multiple options params
            elif isinstance(value, list):
                model_params[key] = st.selectbox(label=key, options=value)
            # skip random state, value will be fixed when creating object
            elif key=='random_state':
                    pass
            # text params
            else:
                model_params[key] = st.text_input(label=key, value=value)
                if model_params[key]=='None':
                    model_params[key]=None
    
    return model_params



######################################################
#                       Main
######################################################

st.session_state
estimator_options = ('LogisticRegression', 
                    'RandomForestClassifier', 
                    'GradientBoostingClassifier', 
                    'AdaBoostClassifier')
# Title and Subheader
st.title("ML Interpreter")
st.subheader("Classifiers visually explained")

st.sidebar.header('Settings')

# Select dataset
dataset_name = st.sidebar.selectbox('Select a dataset:', options=('iris', 'penguins', 'titanic'))

# Read dataset
df, target_name  = select_dataset(dataset_name)
st.sidebar.markdown(f'Target: `{target_name}`')
if st.sidebar.checkbox('Data frame preview'):
    st.subheader(f'{dataset_name} dataset')
    st.dataframe(df.head())

# Select estimator
estimator = st.sidebar.selectbox('Select your model', options=estimator_options)

# Open sidebar with estimator params
model_params = configure_estimator_params(eval(estimator))

# Prepare an clean data before modeling
X, y, target_labels = prepare_data(df, target_name)
# split into train/test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)
# create model
model = eval(estimator)(**model_params, random_state=42)

# Run Model
if st.sidebar.button('Run model'):
    # fit model
    model.fit(X_train, y_train)
    st.success('Pau no gato')     






st.button('Whatever', help='Visit [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html) for parameter details')

