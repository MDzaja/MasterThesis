import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf
import pickle

from weights import backward_looking as bl
from weights import forward_looking as fl
from weights import sequential_return as sr
from weights import trend_interval_return as tir

def plot_weights(title: str, prices: pd.Series, labels: pd.Series, weights: pd.Series):
    # Get union of indices
    union_index = prices.index.union(labels.index).union(weights.index)
    # Reindex all series to this union of indices
    prices_ = prices.reindex(union_index)
    labels_ = labels.reindex(union_index)
    weights_ = weights.reindex(union_index)
    
    fig, ax1 = plt.subplots(figsize=(15, 5))

    # Plot prices on ax1
    ax1.plot(union_index, prices_, color='black', label='Asset price')
    ax1.set_ylabel('Price', color='black')
    ax1.set_title(title)
    ax1.tick_params(axis='y', labelcolor='black')
    # Format x-axis to only show time
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=None))

    # Create the third axis for labels on the left side
    ax3 = ax1.twinx()
    # Move ax3 to the left
    ax3.spines['left'].set_position(('outward', 60))
    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')
    # Plot labels on ax3
    ax3.plot(union_index, labels_, color='orange', linestyle='-', marker='o', label='Trend labels')
    ax3.set_ylabel('Label', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3.set_yticks([1, 0])

    # Create the second axis for weights
    ax2 = ax1.twinx()
    # Plot weights on ax2
    ax2.plot(union_index, weights_, color='blue', label='Weights', linestyle='-', marker='.')
    ax2.set_ylabel('Weights', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Adjusting legend
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc=0)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'

    prices = yf.download(ticker_symbol, start_date, end_date, interval='1d')['Close']
    prices.index = prices.index.tz_localize(None)

    with open('../artifacts/labels/labels_dict_2000-2023_w_23y_params.pkl', 'rb') as file:
        labels_dict = pickle.load(file)
    labels_dict = {k: pd.Series(v, index=pd.to_datetime(v.index)) for k, v in labels_dict.items()}

    all_weights = {}
    for label_name, labels in labels_dict.items():
        all_weights[label_name] = {}
        all_weights[label_name]['backward_looking'] = bl.get_weights(prices, labels)
        all_weights[label_name]['forward_looking'] = fl.get_weights(prices, labels)
        all_weights[label_name]['sequential_return'] = sr.get_weights(prices, labels)
        all_weights[label_name]['trend_interval_return'] = tir.get_weights(prices, labels)

    with open('../artifacts/weights/weights_2000-2023_w_lbl_23y_params.pkl', 'wb') as file:
        pickle.dump(all_weights, file)
