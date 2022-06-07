######################################################
#                   Functions
######################################################

# For Docstrings
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

# Machine Learning with Sklearn
## Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder, label_binarize
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

# Create class to drop columns, used in feature engineering pipeline
class ColumnDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns_to_drop):
        
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        
        return self 

    def transform(self, X, y=None):
        
        return X.drop(columns = self.columns_to_drop)

# Control train/test slider
def train_to_test():
    st.session_state.test_size = 1 - st.session_state.train_size
def test_to_train():
    st.session_state.train_size = 1 - st.session_state.test_size

# Check if any type of feature engineering was selected
def feature_eng_check(features_creator, cols_to_drop):

	# Check if there is a transformer
	feat_eng_pipe_params = []
	if features_creator:
		feat_eng_pipe_params.append( ('create_surname', features_creator) )
	if cols_to_drop:
		feat_eng_pipe_params.append( ('column_dropper', ColumnDropper(cols_to_drop)) )
	
	# Has at least 1 transformer
	if len(feat_eng_pipe_params) > 0:
		return feat_eng_pipe_params
	# No transformer was passed
	else:
		return False

# Create feature engineering pipeline and transform dataset
def apply_feature_engineering(feat_eng_pipe_params, y_train, X_train, X_test):

		print('Applying feature engineering...')
		feature_eng_pipeline  = Pipeline(feat_eng_pipe_params).fit(X_train, y_train)
		
		# Transform features
		X_train = feature_eng_pipeline.transform(X_train)
		X_test = feature_eng_pipeline.transform(X_test)

		return X_train, X_test

# Create preprocess pipeline
def create_preprocess_pipeline(X_train, numeric_params, categorical_params):
    # Define numeric/categorical features
    numeric_features     = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    pipeline = []

    # Create Column transformer with respective parameters
    if len(numeric_features): # No numerical features on dataframe
        if categorical_params: # has transformer
            pipeline.append( ('categorical_transformer', Pipeline(categorical_params) ,categorical_features) )
            return ColumnTransformer(pipeline)
    elif len(categorical_features): # No categorical features on dataframe
        if numeric_params: # has transformer
            pipeline.append( ('numeric_transformer', Pipeline(numeric_params), numeric_features) )
            return ColumnTransformer(pipeline)
    else: # Both types of features and transformers
        if numeric_params:
            pipeline.append( ('numeric_transformer', Pipeline(numeric_params), numeric_features) )
        if categorical_params:
            pipeline.append( ('categorical_transformer', Pipeline(categorical_params) ,categorical_features) )
        if len(pipeline):
            return ColumnTransformer(pipeline)
    # no transformers
    return None

# Create final pipeline and fit model
def create_pipeline(X, y, pp_pipeline, estimator, default_params={}, multi_class=False):
		
    start_time =  datetime.now()
    print(f'Fitting model: ')
    if pp_pipeline:
        pipeline = Pipeline([
            ('pre_processing', pp_pipeline),
            ('estimator', estimator(**default_params))
        ])
    else:
        if multi_class: # for categorical target with more than 2 classes
            pipeline = Pipeline([
                ('estimator', OneVsRestClassifier(estimator(**default_params)))
            ])
        else:
            pipeline = Pipeline([
                ('estimator', estimator(**default_params))
            ])

    pipeline.fit(X, y)
    end_time = datetime.now()
    print(f'Time to fit model: ', str(end_time - start_time).split(".")[0])

    return pipeline

# Run every step of the workflow after all parameters were chosen
def run_model(df:str, target_name:str, estimator:Any, metric_type:str,
			numeric_pipeline:list[Tuple[str, Any]], categorical_pipeline:list[Tuple[str, Any]], 

			train_size:float=0.8, test_size:float=0.2,
			estimator_params:dict={}, stratify:bool=False, multi_class=False,
			eval_df:Optional[str]=None, id_column:Optional[str]=None,
			features_creator:Optional[Any]=None, cols_to_drop:Optional[list[str]]=None, 
			plot_metrics:bool=True, save_model:bool=False, submit_file:bool=False, random_state=42):

    # Set Features
    X = df.drop(columns=target_name) 
    # Set Target
    y = df[target_name]				 
    
    # Check stratify
    if stratify: stratify = y
    else: stratify = None

    # Create target labels
    target_labels = dict( enumerate(y.astype('category').cat.categories ) )

    # Create split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=train_size, 
                                                    test_size=test_size, 
                                                    stratify=stratify, 
                                                    random_state=random_state)
    print(f'Train dataset size: {X_train.shape}')
    print(f'Test dataset size: {X_test.shape}')

    # Feature Engineering
    feat_eng_pipe_params = feature_eng_check(features_creator, cols_to_drop)
    if feat_eng_pipe_params:
        X_train, X_test = apply_feature_engineering(feat_eng_pipe_params, y_train, X_train, X_test)
        
    # Create Pre-processing Pipeline
    pre_processing_pipeline = create_preprocess_pipeline(X_train=X_train,
                                                    numeric_params=numeric_pipeline,
                                                    categorical_params=categorical_pipeline)
    # Make pipeline
    pipeline = create_pipeline(X=X_train, y=y_train, 
                            pp_pipeline=pre_processing_pipeline, 
                            estimator=estimator, default_params=estimator_params,
                            multi_class=multi_class,random_state=random_state)
    # Fit model
    pipeline.fit(X_train, y_train)

    # Success
    return {
        'pipeline': pipeline,
        'X' : X, 'y' : y,
        'X_train' : X_train, 'X_test' : X_test, 
        'y_train' : y_train, 'y_test' : y_test,
        'target_labels' : target_labels
    }

def print_regression_metrics(y_train, y_test, y_pred_train, y_pred_test):
	with st.container():
		st.markdown('**Train metrics**')
		for metric in [r2_score, mean_absolute_error, mean_squared_error]:
			st.text(f'{metric.__name__}: {metric(y_train, y_pred_train):.3f}')
		st.markdown('**Test metrics**')
		for metric in [r2_score, mean_absolute_error, mean_squared_error]:
			st.text(f'{metric.__name__}: {metric(y_test, y_pred_test):.3f}')

######################################################
#             Data Engineering Functions
######################################################

# Extract a short docstring, all characters between the Parameters and Attributes section
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

# Extract all values for params with multiple options
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

# Returns a dictionary with all parameters set with default options
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
#             Input/Output Functions
######################################################

# Read sample data
def read_sample_data(dataset_name):

    if dataset_name == 'iris':
        target_name = 'species'
    elif dataset_name == 'penguins':
        target_name = 'species'
    elif dataset_name == 'diamonds':
        target_name = 'cut'
    elif dataset_name == 'tips':
        target_name = 'sex'

    return prepare_sample_data(dataset_name, target_name)

# Prepare sample data to modeling
def prepare_sample_data(dataset_name, target_name, add_noise=True):
    """  
    \nPreprocess data\n---\n
    Apply every transformation need in order to fil models, like onehot encoding and fill null values\n
    - df : `pd.DataFrame`, used to apply changes  
    - target_name : `str`, column to be predicted  
    \nReturns\n---\n
    - X :  `pd.Dataframe`, transformed features
    - y :  `pd.Series`, target values
    - target_labels : `dict`, mapping of categorical values 
    \nExample\n---\n
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> df = sns.load_dataset('iris')
    >>> X, y, target_labels = prepare_data(df, target_name='species')
    """

    # Read csv with pandas
    df = pd.read_csv(f'sample_data/' + dataset_name + '.csv')

    # Drop target column
    X = df.drop(target_name, axis=1)

    # Add noisy features to make the problem harder
    if add_noise:
        X_noise = X.select_dtypes(include=(int, float)).copy()
        np.random.seed(42)
        mu, sigma = 0, 5
        noise = np.random.normal(mu, sigma, [X_noise.shape[0], X_noise.shape[1]]) 
        X = pd.concat([
                        X.select_dtypes(exclude=(int, float)),              # columns with object, str etc
                        X.select_dtypes(include=(int, float)) + noise     # numeric columns
                    ],
                    axis=1)

    # Onehot encoding for categorical features, and fill null values
    X = pd.get_dummies(X).fillna(0)
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

    # Split into train/test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

    # Join X an y
    df = pd.concat([y, X], axis=1)
    df.rename(columns={0:target_name}, inplace=True)

    # Build dictionary with all variables to return
    results = {
        'X' : X, 'y' : y, 
        'target_labels' : target_labels,
        'target_name' : target_name,
        'X_train' : X_train, 'X_test' : X_test, 
        'y_train' : y_train, 'y_test' : y_test,
        'df' : df
    }

    return results  

# Read file from user input
def read_upload_file(file):
    # read with pandas
    df = pd.read_csv(file, encoding='utf-8')
    # reorders the last column to the second position (target is usually in the first or last column)
    column_selector = df.columns[:-1].insert(1, df.columns[-1])
    return df, column_selector


######################################################
#             Machine Learning Functions
######################################################

# Generate a Plotly figure to plot Confusion Matrix
def create_confusion_matrix(y_true, y_pred, target_labels, name):

    # Calculate confusion matrix
    data = confusion_matrix(y_true, y_pred)

    # Get values from target_labels
    labels = list(target_labels.values())

    # Create PX figure object
    fig = px.imshow(data,
                    labels=dict(
                        x='Predicted label',
                        y='True label',
                        color='# Predictions'),
                    x=labels, y=labels,
                    color_continuous_scale='RdBu_r',
                    text_auto=True
                    )
    
    # Update changes
    fig.update_layout(
                    title={ 
                        'text' : f'Confusion Matrix for {name} dataset',
                        'xanchor' : 'center',
                        'x' : 0.5
                    })
    fig.update_xaxes(side='bottom')

    # Return PX figure to st.plotly_chart()
    return fig

# Generate a Plotly figure to plot ROC curve for binary classification
def plot_binary_roc_auc(y_true, y_score):
        
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Draw area under the curve
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    # Add curve line
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    # Additional customization
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.update_xaxes(constrain='domain')

    return fig

# Calculate metrics for multiple classification problem
def calculate_roc_auc_multiclass(y_true, y_scores, model):
    '''
    Calculate FPR, TPR and ROC AUC score for a multiclass problem\n
    
    \nReturn\n---\n 
    - fpr : False positive rate  
    - tpr : True positive rate  
    - roc_auc : ROC AUC Score  
    '''
    # Encode labels
    y_onehot = label_binarize(y_true, classes=model.classes_)
    # Number of classes
    n_classes = len(model.classes_)
    # dictionary instances
    fpr = dict(); tpr = dict(); roc_auc = dict();
    # Get FP/TP and ROC AUC Score for each class
    for i in range(y_scores.shape[1]):
        y_true = y_onehot[:, i]
        y_score = y_scores[:, i]
        # false positives and true positives
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        # roc auc score
        roc_auc[i] = roc_auc_score(y_true, y_score)

    # Compute micro-average ROC curve and ROC area from prediction scores
    fpr['micro'], tpr['micro'], _ = roc_curve(y_onehot.ravel(), y_scores.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return fpr, tpr, roc_auc

# Generate a Plotly figure to plot ROC curve for multiclass
def plot_multiclass_roc_auc(y_true, y_scores, model, target_labels):

    # get values
    fpr, tpr, roc_auc = calculate_roc_auc_multiclass(y_true, y_scores, model)
    labels = model.classes_
    # Create an empty figure, and iteratively add new lines
    fig = go.Figure()
    fig.add_shape(type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1)
    
    # Add new line for each class
    for label_index in labels:
        name = f'{target_labels[label_index]} (AUC={roc_auc[label_index]:.2f})'
        fig.add_trace(go.Scatter(x=fpr[label_index], y=tpr[label_index], name=name, mode='lines'))
        
    # Customize layout
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )

    return fig

# This is the 'main metrics' function, which calls all the above
# Calculate all metrics and figure objects for classification problem (binary/multiclass)
def calculate_metrics(X, y_true, model, target_labels, split_type):

    # Predictions
    y_pred = model.predict(X)

    # Proba scores, ROC AUC score, F1 score, ROC curve
    # Binary classification
    if len(target_labels) < 3: 
        y_proba = model.predict_proba(X)[:,1]
        roc_auc_score_ = roc_auc_score(y_true, y_proba, multi_class="raise")
        f1_score_ = f1_score(y_true, y_pred, average="binary")
        roc_curve_fig = plot_binary_roc_auc(y_true, y_proba)
    # Multiclass
    else:   
        y_proba = model.predict_proba(X)
        roc_auc_score_ = roc_auc_score(y_true, y_proba, multi_class="ovr")
        f1_score_ = f1_score(y_true, y_pred, average="weighted")
        roc_curve_fig = plot_multiclass_roc_auc(y_true, y_proba, model, target_labels)
    
    # Confusion Matrix
    cf_matrix_fig = create_confusion_matrix(y_true, y_pred, target_labels, name=split_type)

    # Wrap results in a dictionary
    metrics_results = {
        'y_pred' : y_pred,
        'y_proba' : y_proba,
        'roc_auc_score_' : roc_auc_score_,
        'f1_score_' : f1_score_,
        'cf_matrix_fig' : cf_matrix_fig,
        'roc_curve_fig' : roc_curve_fig,
    }
    return metrics_results

# Fit model for sample data
def fit_model(model, X, y):
    try:
        # start timerasas
        start_time =  datetime.now()
        # fit model
        model.fit(X, y)
        # calculate total time
        end_time = datetime.now()
        total_time = f'Time to fit: ' + str(end_time - start_time).split(".")[0]
        st.sidebar.success(total_time)

        return model

    except NotFittedError:
        not_fitted_error()
    except (ValueError, TypeError, AttributeError) as error:
        collapsed_expander_bug()

######################################################
#               Rendering Functions
######################################################

# Show home page
def show_home_page(home_placeholder):

    with home_placeholder.container():

        # Title and Subheader
        st.title("ML Interpreter")
        st.subheader("Classifiers visually explained")
        st.markdown('''
            How to analyze your data:  
            - Upload your csv file, or select a sample data  
            - Define target  
            - Select ID column to drop, if any  
            - Select train/test size and if will stratify target  
            - Apply feature engineering, if necessary  
            - Choose transformers for numerical data  
            - Choose transformers for numerical data  
            - Select estimator to be used  
            - Select hyperparameters
            - Run model, and check metrics score 
        ''')

# Settings to select
# - Target name
# - Encode target label
# - Apply Features Creator
# - Column ID to drop
def target_features_settings(column_selector):

    with st.sidebar.expander('Target and Feature Engineering options'):
        
        # Select target
        target_name = st.selectbox('Choose the target variable', column_selector, key='target_name')
            
        # Select to encode target (categorical data)
        target_encode = st.checkbox(label='Encode target', 
                                    help='Use LabelEncode to process categorical values', 
                                    key='target_encode')

        # Features Creator
        fc_check = st.checkbox('Apply FeaturesCreator')
        if fc_check:
            feature_creator = st.file_uploader('Upload a FeaturesCreator object', key='features_creator')
        else:
            feature_creator = None

        # Columns to drop
        if st.checkbox('Drop columns'):
            cols_to_drop = st.multiselect('Select columns to drop', 
                                        options=list(column_selector).remove(target_name), 
                                        key='cols_to_drop')
        else:
            cols_to_drop = []
    
    return {
        'target_name' : target_name,
        'target_encode' : target_encode,
        'fc_check' : fc_check,
        'feature_creator' : feature_creator,
        'cols_to_drop' : cols_to_drop
    }

# Split data into train/test
# - Select train size
# - Select test size
# - Select if stratify target
def test_train_split():

    # Train/Test Split parameters
    with st.sidebar.expander('Train/Test Split parameters'):
        train_size = st.slider('Train size', min_value=0.05, max_value=0.95, on_change=train_to_test, key='train_size')
        test_size = st.slider('Train size', min_value=0.05, max_value=0.95, on_change=test_to_train, key='test_size')
        stratify = st.checkbox('Stratify target')
    
    return {
        'train_size' : train_size,
        'test_size' : test_size,
        'stratify' : stratify
    }

# Select transformers for numerical data
def numerical_transformer():

    with st.sidebar.expander('Transformers for Numerical Features'):
        numeric_pipeline = []
        # Numerical Imputer
        if st.checkbox('Imputer', key='num_imputer'):
            imputer_strategy = st.selectbox('Select strategy:', options=('mean', 'median', 'most_frequent'))
            numeric_pipeline.append( ('impute_num', SimpleImputer(strategy=imputer_strategy)) )
        # Numerical Scaler
        scaler = st.radio('Scale transformer', options=(None, 'StandardScaler', 'MinMaxScaler'), key='num_scaler')
        if scaler == 'StandardScaler':
            numeric_pipeline.append( ('std', StandardScaler()) )
        elif scaler == 'MinMaxScaler':
            numeric_pipeline.append( ('mms', MinMaxScaler()) )
    
    return {
        'numeric_pipeline' : numeric_pipeline
    }

# Select transformers for categorical data
def categorical_transformer():
    # Transformers for Categorical Features
    with st.sidebar.expander('Transformers for Categorical Features'):
        categorical_pipeline = []
        # Categorical Imputer
        if st.checkbox('Imputer', key='cat_imputer'):
            st.text("Imputer strategy = 'constant'")
            fill_value = st.text_input(label='fill value', help="default value = 'unknow'")
            if not fill_value:
                fill_value = 'unknow'
            categorical_pipeline.append( ('impute_cat', SimpleImputer(strategy='constant', fill_value=fill_value)) )
        # Variable Encoding
        encoder = st.radio('Encoder', options=(None, 'OneHotEncoder', 'OrdinalEncoder'), key='cat_endocer')
        if encoder == 'OneHotEncoder':
            categorical_pipeline.append( ('onehot', OneHotEncoder(handle_unknown='ignore')) )
        elif encoder == 'OrdinalEncoder':
            categorical_pipeline.append( ('ordinal', OrdinalEncoder()) )
    
    return {
        'categorical_pipeline' : categorical_pipeline
    }

# Display a summary of all options selected
def options_summary(target_name, test_size, train_size, stratify, fc_check, cols_to_drop, numeric_pipeline, categorical_pipeline, **kwargs):
    
    with st.sidebar.expander('Parameters summary'):
        st.markdown(f'**Target**: {target_name}')
        st.markdown(f'**Test/train size**: {test_size:.2f} / {train_size:.2f}')
        st.markdown(f'**Stratify**: {stratify}')
        st.markdown(f'**Feature Creator**: {fc_check}')
        st.markdown(f'**Drop columns**: {", ".join(cols_to_drop)}')
        st.markdown(f'**Numerical Transformers**: {", ".join([str(transformer[-1]) for  transformer in numeric_pipeline])}')
        st.markdown(f'**Categorical Transformers**: {", ".join([str(transformer[-1]) for  transformer in categorical_pipeline])}')

# Show estimator parameters to choose in Sidebar
def configure_estimator_params(estimator):

    model_params = {}

    with st.sidebar.expander('Configure parameters'):
        for key, value in get_default_params(estimator).items():
            # True or False params
            if isinstance(value, bool):
                model_params[key] = st.checkbox(label=key, value=value)
            # float or int params
            elif isinstance(value, (int,float)):
                model_params[key] = st.number_input(label=key, value=value, help=str(type(value)))
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

# Create full pipeline for uploaded file
def build_pipeline(df:str, target_name:str, estimator:Any,
			numeric_pipeline:list[Tuple[str, Any]], categorical_pipeline:list[Tuple[str, Any]], 

			train_size:float=0.8, test_size:float=0.2, target_encode=False,
			hyper_params:dict={}, stratify:bool=False, multi_class=False,
			features_creator:Optional[Any]=None, cols_to_drop:Optional[list[str]]=None, 
			random_state=42, **kwargs):

    # Set Features
    X = df.drop(columns=target_name) 
    # Set Target
    y = df[target_name]				 
    
    # Check stratify
    if stratify: stratify = y
    else: stratify = None

    # Create target labels
    target_labels = dict( enumerate(y.astype('category').cat.categories ) )

    # Create split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=train_size, 
                                                    test_size=test_size, 
                                                    stratify=stratify, 
                                                    random_state=random_state)
    print(f'Train dataset size: {X_train.shape}')
    print(f'Test dataset size: {X_test.shape}')

    # Feature Engineering
    feat_eng_pipe_params = feature_eng_check(features_creator, cols_to_drop)
    if feat_eng_pipe_params:
        X_train, X_test = apply_feature_engineering(feat_eng_pipe_params, y_train, X_train, X_test)
        
    # Create Pre-processing Pipeline
    pre_processing_pipeline = create_preprocess_pipeline(X_train=X_train,
                                                    numeric_params=numeric_pipeline,
                                                    categorical_params=categorical_pipeline)
    # Make pipeline
    pipeline = create_pipeline(X=X_train, y=y_train, 
                            pp_pipeline=pre_processing_pipeline, 
                            estimator=estimator, default_params=hyper_params,
                            multi_class=multi_class)

    return {
        'pipeline': pipeline,
        'X' : X, 'y' : y,
        'X_train' : X_train, 'X_test' : X_test, 
        'y_train' : y_train, 'y_test' : y_test,
        'target_labels' : target_labels
    }

# Display Metrics summary and plot confusion matrix/roc auc curve
# This is functions is called to display it's values in a st.column
def plot_metrics(roc_auc_score_, f1_score_, cf_matrix_fig, roc_curve_fig, **kwargs):

    # Write metrics
    st.text(f'ROC AUC Score = {roc_auc_score_:.3f}')
    st.text(f'F1 Score = {f1_score_:.3f}')
    # Plot figures
    st.plotly_chart(cf_matrix_fig, use_container_width=True)
    st.plotly_chart(roc_curve_fig, use_container_width=True)

# Main function to calculate and display metrics
def display_metrics(model, X_train, X_test, y_train, y_test, target_labels,**kwargs):

    # Calculate metrics for Train dataset
    train_metrics = calculate_metrics(X_train, y_train, model, target_labels, split_type='Train')
    # Calculate metrics for Train dataset
    test_metrics = calculate_metrics(X_test, y_test, model, target_labels, split_type='Test')
    # Display results
    col1, col2 = st.columns(2)
    # Train column
    with col1:
        st.markdown('**Train metrics:**')
        plot_metrics(**train_metrics)
    # Test column
    with col2:
        st.markdown('**Test metrics:**')
        plot_metrics(**test_metrics)


######################################################
#                      Errors
######################################################

# Handling NotFittedError message
def not_fitted_error():
    st.markdown('''
            **Error:** Model is not fitted!  
            This can happen if you change some settings from sidebar after model is complete.  
            *Plase run again your model in the sidebar button.*  
            ''')

# Handling bug with collapsed expander
def collapsed_expander_bug():
    st.markdown('''There is a bug when running some estimators for the first time with the  
    **"Configure parameters"** expander collapsed. Just open/close the expander and run again.  
    ''')
