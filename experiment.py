'''
This file provides an Experiment object containing common methods for forecast experiments.
'''
import pandas as pd
from utils import totimestamp, makelist
from features import FeatureFactory

# pylint: disable=too-many-instance-attributes
class Experiment:
    '''
    Experiment class which encapsulates all configurations of a forecasting experiment.
    '''
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        init_data: pd.DataFrame,
        history_start: str,
        forecast_start: str,
        forecast_end: str,
        timestamp_col: str,
        attribute_cols: 'list[str]',
        feature_cols:'list[str]',
        target_col: str,
        ):
        # settings of experiment
        self.history_start = history_start
        self.forecast_start = forecast_start
        self.timestamp_col = timestamp_col
        self.forecast_end = forecast_end
        self.attribute_cols = attribute_cols
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.data = self._generate_data(init_data)

        # objects of the experiment
        self.factory = FeatureFactory(
            self.data,
            forecast_start,
            timestamp_col,
            attribute_cols,
            feature_cols,target_col
            )

    def _generate_data(self, init_data: pd.DataFrame) -> pd.DataFrame:
        timestamp_col = self.timestamp_col
        init_data = init_data[
            (init_data[timestamp_col]>=totimestamp(self.history_start)) &
            (init_data[timestamp_col]<totimestamp(self.forecast_end))
            ]
        return init_data

    def get_train_test(self, data: pd.DataFrame):
        '''
        Splits data into train and test set.
        '''
        timestamp_col = self.timestamp_col
        train = data[
            (data[timestamp_col]>=totimestamp(self.history_start)) &
            (data[timestamp_col]<totimestamp(self.forecast_start))
            ]
        test = data[
            (data[timestamp_col]>=totimestamp(self.forecast_start)) &
            (data[timestamp_col]<totimestamp(self.forecast_end))
            ]
        return train, test

    def get_x_y(self, data, **attrubute_values):
        '''
        Splits dat into train features, test features and train target values.
        '''
        train, test = self.get_train_test(data)
        cols = list(attrubute_values.keys())
        vals = [makelist(attrubute_values[col]) for col in cols]
        for col, val in zip(cols, vals):
            train = train[train[col].isin(val)]
            test = test[test[col].isin(val)]

        x_train, x_test = train[self.feature_cols], test[self.feature_cols]
        y_train = train[self.target_col]
        return x_train, x_test, y_train

    def compute_metric(self):
        '''
        Computes metrics (accuracy/error) of forecast for both train and test set.
        '''
        print()

    def cross_calidation(self):
        '''
        Does cross-validation for the forecast.
        '''
        print()
    