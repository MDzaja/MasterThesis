import sys
sys.path.insert(0, '../')

from backtesting import Strategy
import numpy as np
import pandas as pd

from models import utils as model_utils


def ModelBasedIntradayStrategyFactory(model, window, scale_reference_data=None):
    class ModelBasedIntradayStrategy(Strategy):
        def init(self):
            super().init()
            self.model = model
            self.window = window
            self.scale_reference_data = scale_reference_data
            self.predictions = self.I(self.make_predictions, self.data.df)
            self.whole_data = self.data.df.copy()

        def make_predictions(self, data):
            predictions = []

            for date, group in data.groupby(pd.Grouper(freq='D')):
                if len(group) < self.window:
                    predictions.extend([0]*len(group))
                else:
                    predictions.extend([0]*self.window)
                    predictions.extend(self.model.predict(model_utils.get_X(group, self.window, self.scale_reference_data), verbose=0).flatten())

            return predictions

        def next(self):
            # Close position at the end of the day
            current_index = len(self.data) - 1
            if current_index < len(self.whole_data) - 1:
                if self.data.index[-1].date() != self.whole_data.index[current_index + 1].date():
                    if self.position:
                        self.position.close()
                        return

            # Trading logic
            if self.predictions[-1] > 0.5:
                if not self.position:
                    self.buy()
            else:
                if self.position:
                    self.position.close()

    return ModelBasedIntradayStrategy


def PredictionsIntradayStrategyFactory(probs: pd.Series):
    class PredictionsIntradayStrategy(Strategy):

        threshold=0.5

        def init(self):
            super().init()
            self.init_probs = probs
            self.predictions = self.I(self.make_predictions)
            self.whole_data = self.data.df.copy()

        def make_predictions(self):
            return self.init_probs.values

        def next(self):
            # Close position at the end of the day
            current_index = len(self.data) - 1
            if current_index < len(self.whole_data) - 1 and self.data.index[-1].date() != self.whole_data.index[current_index + 1].date():
                if self.position:
                    self.position.close()
            # Trading logic
            elif self.predictions[-1] > self.threshold:
                if not self.position:
                    self.buy()
            else:
                if self.position:
                    self.position.close()

    return PredictionsIntradayStrategy


def BuyAndHoldStrategyFactory(window):
    class BuyAndHoldStrategy(Strategy):
        def init(self):
            self.window = window
            self.has_bought = False

        def next(self):
            # Check if the waiting period has passed and if we haven't bought the asset yet
            if len(self.data) >= self.window and not self.has_bought:
                self.buy()
                self.has_bought = True

    return BuyAndHoldStrategy
