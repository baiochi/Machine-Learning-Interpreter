from collections import namedtuple
from faulthandler import disable
import altair as alt
import math
from isort import file

from typing import Union, Optional, Tuple, Any
from datetime import datetime
import time
import pickle
import base64

from requests_cache import disabled
from functions import *

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px

# Machine Learning - Scikit-Learn
import sklearn

# Prior Modeling
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, \
                            RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,\
	ConfusionMatrixDisplay, classification_report, roc_auc_score, f1_score, roc_curve, plot_roc_curve, precision_recall_curve
from xgboost import XGBClassifier, XGBRegressor

######################################################
#                Page Configuration
######################################################
# Brownser tab title and icon
st.set_page_config(
        page_title = 'ML App',
        page_icon = ':potato:',
        menu_items = {
            'About': '''
            **Author**: João Baiochi  
            [Github](https://github.com/baiochi)  
            [E-mail](mailto:joao.baiochi@outlook.com.br)
            '''
        }
)
# Session state innit
#st.session_state
model_results = None

# Load CSS file
# For change defaul theme, create config.toml file
# streamlit config show > ~/.streamlit/config.toml
# add options into [theme section]
#with open('style.css') as file:
#    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

######################################################
#                   Main Title
######################################################

guide_placeholder = st.empty()
with guide_placeholder.container():
    st.title('> Machine Learning Playground')
    st.caption('Modeling examples with [`Scikit-Learn`](https://scikit-learn.org/stable/) library')
    st.subheader('Workflow guide:')
    st.markdown('''
        - Upload your csv file  
        - Define target  
        - Select ID column to drop, if any  
        - Select train/test size and if will stratify target  
        - Apply feature engineering, if necessary  
        - Choose transformers for numerical data  
        - Choose transformers for numerical data  
        - Select estimator to be used  
        - Run model, and check metrics score  
    ''')

# Sidebar Configuration
st.sidebar.header('Start here')
with st.sidebar.expander('Upload a CSV file', expanded=True):
    file_upload = st.file_uploader('', type='csv', key='file_loader')
    if file_upload:
        guide_placeholder.empty()

######################################################
#              Run when file is loaded
######################################################
if file_upload:
    
    # Load dataframe
    df, column_selector, id_selector  = read_dataframe(file_upload)

    # Dataframe preview
    if st.sidebar.checkbox('Show dataframe preview', key='data_prev'):
        st.subheader('Dataframe Preview')
        st.dataframe(df.iloc[np.r_[0:3, -3:0]]) # show head and tail

    st.sidebar.header('Select parameters to run model')

    # Define Target variable
    with st.sidebar.expander('Define Target and Features'):
        # Select target
        target = st.selectbox('Choose the target variable', column_selector, key='target')
        # Select column to drop (Optional)
        id_selector.remove(target)
        id_column = st.selectbox('Choose the ID column to drop', id_selector, key='id_column')
        if id_column:
            df.drop(columns=id_column, inplace=True)
        # Select to encode target (categorical data)
        target_encode = st.checkbox(label='Endoce target', 
                                    help='Use LabelEncode to process categorical values', 
                                    key='target_encode')

    # Train/Test Split parameters
    with st.sidebar.expander('Train/Test Split parameters'):
        train_size = st.slider('Train size', min_value=0.05, max_value=0.95, on_change=train_to_test, key='train_size')
        test_size = st.slider('Train size', min_value=0.05, max_value=0.95, on_change=test_to_train, key='test_size')
        stratify = st.checkbox('Stratify target')
    
    # Feature Engineering
    with st.sidebar.expander('Feature Engineering'):
        fc_check = st.checkbox('Apply FeaturesCreator')
        if fc_check:
            feature_creator = st.file_uploader('Upload a FeaturesCreator object', key='features_creator')
        else:
            feature_creator = None
        if st.checkbox('Drop columns'):
            cols_to_drop = st.multiselect('Select columns to drop', 
                                        options=df.drop(columns=target).columns, 
                                        key='cols_to_drop')
        else:
            cols_to_drop = []

    # Transformers for Numerical Features
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
        
    # Estimator
    with st.sidebar.expander('Select Estimator'):
        learning_type = st.radio('Problem type', options=('Regression', 'Classification'))
        if learning_type == 'Regression':
            estimator = st.selectbox('Options', options=('LinearRegression',
                                                'RandomForestRegressor',
                                                'SVR', 
                                                'XGBRegressor')
                                    )
        elif learning_type == 'Classification':
            estimator = st.selectbox('Options', options=('LogisticRegression',
                                                'RandomForestClassifier',
                                                'SVC',
                                                'XGBClassifier')
                                        )

    # Summary
    with st.sidebar.expander('Parameters summary'):
        st.markdown(f'**Target**: {target}')
        st.markdown(f'**Drop ID**: {id_column}')
        st.markdown(f'**Test/train size**: {test_size:.2f} / {train_size:.2f}')
        st.markdown(f'**Stratify**: {stratify}')
        st.markdown(f'**Feature Creator**: {fc_check}')
        st.markdown(f'**Drop columns**: {", ".join(cols_to_drop)}')
        st.markdown(f'**Numerical Transformers**: {", ".join([str(transformer[-1]) for  transformer in numeric_pipeline])}')
        st.markdown(f'**Categorical Transformers**: {", ".join([str(transformer[-1]) for  transformer in categorical_pipeline])}')
        st.markdown(f'**Estimator**: {estimator}')
    
    # Button to run model
    with st.sidebar.form(key='run_model'):
        submitted = st.form_submit_button('Run model')
        if submitted:
            model_results = run_model(df=df, 
                        target_name=target, 
                        estimator=eval(estimator), # convert to object
                        metric_type=learning_type,
                        numeric_pipeline=numeric_pipeline, 
                        categorical_pipeline=categorical_pipeline, 
                        train_size=train_size, 
                        test_size=test_size,
                        estimator_params={}, 
                        stratify=stratify, 
                        features_creator=feature_creator, 
                        cols_to_drop=cols_to_drop, 
                        plot_metrics=False, save_model=False, submit_file=False, random_state=42)
            st.success('Fit complete!!')
            time.sleep(2)
                
        

# Display model performance after run
if model_results:
    # Extract results
    X_train, X_test, y_train, y_test = model_results['train_test_split']
    model = model_results['pipeline']
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    st.header(f'{estimator} Performance')
    
    c1, c2 = st.columns(2)

    with c1.container():
        print_regression_metrics(y_train, y_test, y_train_pred, y_test_pred)
        #st.text(f'Test R2 score: {r2_score(y_test, y_test_pred):.2f}')
        #st.text(f'Train R2 score: {r2_score(y_train, y_train_pred):.2f}')

    c2.download_button('Download model', 
                        data=pickle.dumps(model), 
                        file_name=f'{estimator}_model_{datetime.now().strftime("%H_%M_%S")}.pkl')
    c2.download_button('Download metrics', 
                        data=pickle.dumps(model), 
                        file_name=f'{estimator}_metrics_{datetime.now().strftime("%H_%M_%S")}.pkl')
    







#st.session_state


def select_dataset():
    st.sidebar.write()
    choice = st.sidebar.selectbox('Choose an option:', options=('Sample dataset', 'File upload'))

    if choice == 'Sample dataset':
        sample_data = st.sidebar.selectbox(
                        
                        )
        
        if upload_data_button:
            df, target = load_data(sample_data)
            return sample_data, df, target
    elif choice == 'File upload':
        upload_file = st.sidebar.file_uploader('Or upload a CSV file', type='csv')
        upload_data_button = st.sidebar.button('Load data')
        if upload_data_button:
            df, target = load_data(sample_data=None, upload_file=upload_file)
            return upload_file.split('/')[-1], df, target

    return None, any, any




def load_data(sample_data, upload_file=None):

    # Upload option
    if upload_file:

        st.sidebar.write('Loading file...')
        df = pd.read_csv(upload_file, encoding='utf8')
        st.sidebar.success('Upload successful')
        # reorders the last column to the second position (target is usually in the first or last column)
        column_selector = df.columns[:-1].insert(1, df.columns[-1])
        # create sliders to select column
        target = st.sidebar.selectbox('Choose the target variable', column_selector)
    
    # Sample datasets
    elif sample_data == 'penguins':
        df = sns.load_dataset('penguins')
        target = 'species'

    elif sample_data == 'iris':
        df = sns.load_dataset('iris')
        target = 'species'
    
    elif sample_data == 'tips':
        df = sns.load_dataset('tips')
        target = 'total_bill'

    elif sample_data == 'titanic':
        df = sns.load_dataset('titanic').drop(['class', 'who', 'adult_male', 'deck', 'alive', 'alone'], axis=1)
        target = 'survived'

    return df, target


# click = st.button(
#     'Hello Ezequiel',
#     help='Fuck you!',
#     disabled=False
# )
# st.write(f'Button status: {click}')


# def page_intro():
#     st.header('Batata')
#     st.markdown(':potato:')

# def page_df():
#     st.header('Dataframe')
#     st.dataframe(df)

# def page_plot():
#     st.header('Plot data')
#     figure = px.scatter(
#             df,
#             x='bill_length_mm',
#             y='bill_depth_mm',
#             color='species',
#         )
#     st.plotly_chart(figure)

# pages = {
#     'Intro' : page_intro,
#     'Data overview' : page_df,
#     'Plot data' : page_plot,
# }




# Main function
#def main():

    # file_select, df, target = select_dataset()

    # if file_select:
        
    #     st.write('Fuck you mathias')
    #     selected_page = st.selectbox('Choose page:', pages.keys())

    #     pages[selected_page]()

    #     st.dataframe(df.head())

    #     st.metric(
    #         label='Model Score',
    #         value=0.85,
    #         delta=-0.2
    #     )

    #     if file_select == 'penguins':

    #         figure = px.scatter(
    #             df,
    #             x='bill_length_mm',
    #             y='bill_depth_mm',
    #             color='species',
    #         )

    #         st.plotly_chart(figure)


# Run App
#if __name__ == '__main__':
#    main()



#
#def show_num_lines(dataframe):
#
#    line_number = st.sidebar.slider('Select the number of lines to show in table', 
#                                    min_value = 1, max_value = len(dataframe), step = 1)
#
#    st.write(dataframe.head(line_number).style.format(subset = ['Valor'], formatter='{:.2f}'))
#
#checkbox_show_table = st.sidebar.checkbox('Show table')
#if checkbox_show_table:
#
#    st.sidebar.markdown('## Table filter')
#
#    categories = list(data['Categoria'].unique())
#    categories.append('All')
#
#    category = st.sidebar.selectbox('Select categories to show in table', options = categories)
#
#    if category != 'All':
#        df_categoria = data.query('Categoria == @category')
#        show_num_lines(df_categoria)      
#    else:
#        show_num_lines(data)
#
#
#
#'''
#with st.echo(code_location='below'):
#    total_points = st.slider('Number of points in spiral', 1, 5000, 2000)
#    num_turns = st.slider('Number of turns in spiral', 1, 100, 9)
#
#    Point = namedtuple('Point', 'x y')
#    data = []
#
#    points_per_turn = total_points / num_turns
#
#    for curr_point_num in range(total_points):
#        curr_turn, i = divmod(curr_point_num, points_per_turn)
#        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#        radius = curr_point_num / total_points
#        x = radius * math.cos(angle)
#        y = radius * math.sin(angle)
#        data.append(Point(x, y))
#
#    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#        .mark_circle(color='#0068c9', opacity=0.5)
#        .encode(x='x:Q', y='y:Q'))
#'''