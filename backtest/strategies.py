import sys
sys.path.insert(0, '../')

from backtesting import Strategy
import numpy as np

from models import utils as model_utils


def ModelBasedStrategyFactory(model, window):
    class ModelBasedStrategy(Strategy):
        def init(self):
            super().init()
            self.model = model
            self.window = window
            self.predictions = self.I(self.make_predictions, self.data.df)

        def make_predictions(self, data):
            predictions = self.model.predict(model_utils.get_X(data, self.window))
            predictions = predictions.flatten()
            predictions = np.pad(predictions, (self.window, 0), 'constant')
            return predictions

        def next(self):
            if self.predictions[-1] > 0.5:
                if not self.position:
                    self.buy()
            else:
                if self.position:
                    self.position.close()

    return ModelBasedStrategy


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
