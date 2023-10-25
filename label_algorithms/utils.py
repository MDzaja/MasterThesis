import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt

def plot_labels(title: str, prices: pd.Series, labels: pd.Series):
    # Get union of indices
    union_index = prices.index.union(labels.index)
    # Reindex both series to this union of indices
    prices_ = prices.reindex(union_index)
    labels_ = labels.reindex(union_index)
    fig, ax1 = plt.subplots(figsize=(15, 5))

    # Plot prices on ax1
    ax1.plot(union_index, prices_, color='black', label='Asset price')
    ax1.set_ylabel('Price', color='black')
    ax1.set_title(title)
    ax1.tick_params(axis='y', labelcolor='black')
    # Format x-axis to only show time
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=None))

    # Create the second axis
    ax2 = ax1.twinx()
    # Plot labels on ax2
    ax2.plot(union_index, labels_, color='orange', label='Trend labels', linestyle='-', marker='o')
    ax2.set_ylabel('Label', color='black')
    ax2.set_yticks([1, 0, -1])

    # Show the plot
    plt.show()