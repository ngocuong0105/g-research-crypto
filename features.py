'''
This file provides an object for feature engineering.
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing, decomposition, pipeline, feature_selection
from utils import totimestamp

class FeatureFactory:
    def __init__(
        self, 
        data:pd.DataFrame,
        forecast_start:str, 
        timestamp_col:str,
        attribute_cols: 'list[str]',
        feature_cols: 'list[str]',
        target_col:str
        ) -> None :
        self.data = data
        self.forecast_start = forecast_start
        self.timestamp_col = timestamp_col
        self.attribute_cols = attribute_cols
        self.feature_cols = feature_cols
        self.target_col = target_col

    def clean_features(self) -> pd.DataFrame:
        for _,group in self.data.groupby(self.attribute_cols):
            pipe = pipeline.Pipeline([
                    ('scaler', preprocessing.StandardScaler()),
                    ('pca', decomposition.PCA()),
                    ('filter constants', feature_selection.VarianceThreshold(0.05))
                ])
            train = group[group[self.timestamp_col]<totimestamp(self.forecast_start)]
            Xtrain = train[self.feature_cols]
            ytrain = train[self.target_col]
            pipe.fit_transform(Xtrain,ytrain)
            self.data = pd.DataFrame(
                        pipe.transform(group[self.feature_cols]),
                        index = group[self.feature_cols].index
                        )
            self.feature_cols = self.data.columns

    def add_log_return_feature(self, col_name:'str', periods:int = 10):
        for _,group in self.data.groupby(self.attribute_cols):
            self.data.loc[group.index,col_name] = np.log(group[col_name]).diff(periods=periods)
        
    def add_rolling_mean(self, col_name:'str', windows:int = 10):
        for _,group in self.data.groupby(self.attribute_cols):
            self.data.loc[group.index,col_name + '_mean'] = group[col_name].shift(windows).rolling(windows).mean().fillna(0)
        self.feature_cols.append(col_name + '_mean')
        
    def add_rolling_median(self, col_name:'str', windows:int = 10):
        for _,group in self.data.groupby(self.attribute_cols):
            self.data.loc[group.index,col_name + '_median'] = group[col_name].shift(windows).rolling(windows).median().fillna(0)
        self.feature_cols.append(col_name + '_median')

    def add_rolling_max(self, col_name:'str', windows:int = 10):
        for _,group in self.data.groupby(self.attribute_cols):
            self.data.loc[group.index,col_name + '_max'] = group[col_name].shift(windows).rolling(windows).max().fillna(0)
        self.feature_cols.append(col_name + '_median')