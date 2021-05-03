import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df

        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'),
                            compression='zip', low_memory=False)
        # obtain features from award
        award['Name_processed'] = award['incumbent_name'].str.lower()
        award['Name_processed'] = award['Name_processed'].str.replace('[^\w]','')
        award['End_of_call_date'] = pd.to_datetime(award['End_of_call_date'], format='%Y-%m-%d', errors='coerce')
        award['End_of_call_year'] = award['End_of_call_date'].dt.year
        award_features = award.groupby(['Name_processed'
                                        ,'End_of_call_year'
                                        ])['amount'].agg(['count','sum'])
        
        def process_APE(X):
            #APE1 = X['Activity_code (APE)'].str[:2]
            APE2 = X['Activity_code (APE)'].str[:4]
            return np.c_[
                        #pd.to_numeric(APE1, errors='coerce').values,
                         pd.to_numeric(APE2, errors='coerce').values
                         ]
        APE_transformer = FunctionTransformer(process_APE, validate=False)

        def merge_naive(X):
            X['Name'] = X['Name'].str.lower()     
            X['Name'] = X['Name'].str.replace('[^\w]','')
            df = pd.merge(X, award_features, left_on=['Name'
                                                    ,'Year'
                                                    ], 
                          right_on=['Name_processed'
                                    ,'End_of_call_year'
                                    ], how='left')
            return df[['count','sum']]
        merge_award_transformer = FunctionTransformer(merge_naive, validate=False)

        head_att = ['mean','max','min' ]
        headcount_features = X_encoded.groupby(['Name'])['Headcount'].agg(head_att)
        def merge_headcounts(X):    
            df = pd.merge(X, headcount_features, left_on=['Name'], 
                        right_on=['Name'], how='left')
            return df[head_att]
        merge_head_transformer = FunctionTransformer(merge_headcounts, validate=False)

        def encode_year(X):
            return np.c_[(X['Year']==2013.) * 1.,
                        (X['Year']==2014.) * 1.,
                        (X['Year']==2015.) * 1.,
                        (X['Year']==2016.) * 1.,
                        (X['Year']==2017.) * 1.,
                        (X['Year']==2018.) * 1.]
        encode_year_transformer = FunctionTransformer(encode_year, validate=False)

        num_cols = ['Legal_ID', 'Headcount', 'Year']
        id_col = ['Legal_ID']
        head_col = ['Headcount']
        year_col = ['Year']
        APE_col = ['Activity_code (APE)']
        merge_award_cols = ['Name','Year']
        merge_headcount_col = ['Name']

        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))])

        preprocessor = ColumnTransformer(
            transformers=[
                #('num', numeric_transformer, num_cols),
                ('ID', make_pipeline(SimpleImputer(strategy='median')), id_col),
                ('Headcount', make_pipeline(SimpleImputer(strategy='median')), head_col),
                ('Year', make_pipeline(SimpleImputer(strategy='median')), year_col),
                #('Year encoders', make_pipeline(encode_year_transformer, SimpleImputer(strategy='median')), year_col),
                ('APE', make_pipeline(APE_transformer, SimpleImputer(strategy='median')), APE_col),
                ('merge headcounts', make_pipeline(merge_head_transformer, SimpleImputer(strategy='median')), merge_headcount_col),
                ('merge awards', make_pipeline(merge_award_transformer, SimpleImputer(strategy='median')), merge_award_cols),
                ])

        X_array = preprocessor.fit_transform(X_encoded)

        #print(X_array.shape)

        return X_array