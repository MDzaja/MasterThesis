import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from tabulate import tabulate
from collections import Counter
import numpy as np
from scipy.stats import linregress


def plot_labels(title: str, prices: pd.Series, labels_dict: dict):
    # Calculate the number of rows needed (each row has up to 3 subplots)
    num_labels = len(labels_dict)
    num_rows = (num_labels + 2) // 3  # +2 for rounding up in integer division

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for i, (label_name, labels) in enumerate(labels_dict.items()):
        # Get union of indices
        union_index = prices.index.union(labels.index)
        # Reindex both series to this union of indices
        prices_ = prices.reindex(union_index)
        labels_ = labels.reindex(union_index)

        # Plot prices
        ax1 = axes[i]
        ax1.plot(union_index, prices_, color='black', label='Asset price')
        ax1.set_ylabel('Price', color='black')
        ax1.set_title(f'{title} - {label_name}')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Create the second axis for labels
        ax2 = ax1.twinx()
        ax2.plot(union_index, labels_, color='orange', label='Trend labels', linestyle='-', marker='.')
        ax2.set_ylabel('Label', color='black')
        ax2.yaxis.set_major_locator(MultipleLocator(1))

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_weights(title: str, prices: pd.Series, labels: pd.Series, weights_dict: dict):
    # Calculate the number of rows needed (2 subplots per row)
    num_weights = len(weights_dict)
    num_rows = (num_weights + 1) // 2  # +1 for rounding up in integer division

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))#TODO: change figsize
    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for i, (weight_name, weights) in enumerate(weights_dict.items()):
        # Get union of indices
        union_index = prices.index.union(labels.index).union(weights.index)
        # Reindex all series to this union of indices
        prices_ = prices.reindex(union_index)
        labels_ = labels.reindex(union_index)
        weights_ = weights.reindex(union_index)

        # Plot prices
        ax1 = axes[i]
        ax1.plot(union_index, prices_, color='black', label='Asset price')
        ax1.set_ylabel('Price', color='black')
        ax1.set_title(f'{title} - {weight_name}')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis to show only hours

        # Create the second axis for labels on the left side
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

        # Create the third axis for weights
        ax2 = ax1.twinx()
        # Plot weights on ax2
        ax2.plot(union_index, weights_, color='blue', linestyle='-', marker='.', label='Weights')
        ax2.set_ylabel('Weights', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Adjusting legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc=0)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_roc_auc_curves(true_y_dict, y_prob_dict, title='ROC AUC Curves'):
    """
    Plot the ROC curves for given true labels and multiple sets of probability estimates.

    Parameters:
    true_y_dict (dict): A dictionary where keys are label names and values are the true class values.
    y_prob_dict (dict): A dictionary where keys are model names and values are dictionaries containing 
                        'label_name' (the label name) and 'probs' (the prediction probabilities).

    Returns:
    None: The function will plot the ROC curve for each model.
    """
    plt.figure()

    # Iterate over each model's predictions and plot their ROC curve
    for model_name, model_data in y_prob_dict.items():
        label_name = model_data['label_name']
        y_score = model_data['probs']
        y_test = true_y_dict[label_name]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')

    # Add plot formatting
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    # Move the legend out of the plot, to the right of the plot, centered vertically
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def plot_acc_auc(metrics_dict, x_title, y_title, x_key, y_key, data_types):
    def parse_combination_name(name):
        parts = name.split(';')
        return {p.split('-')[0]: p.split('-')[1] for p in parts}

    def flatten_data():
        flattened_data = []
        for combination, data in metrics_dict.items():
            parsed_name = parse_combination_name(combination)
            for data_type, metrics in data.items():
                if data_type not in data_types:
                    continue
                row = [combination]
                row.extend([parsed_name['D'], parsed_name['M'], parsed_name['L'], parsed_name['W'], data_type])
                row.extend([metrics[x_key], metrics[y_key]])
                flattened_data.append(row)
        return flattened_data

    flattened_data = flatten_data()
    columns = ['Combination', 'Data', 'Model', 'Label', 'Weight', 'Data_Type', x_title, y_title]
    metrics_df = pd.DataFrame(flattened_data, columns=columns)

    # Calculate the number of rows needed based on the length of data_types
    n_rows = (len(data_types) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows), sharex=False, sharey=False)
    if n_rows == 1:  # If there's only one row, axes is a one-dimensional array
        axes = [axes]
    
    for i, data_type in enumerate(data_types):
        ax = axes[i // 2][i % 2]
        subset = metrics_df[metrics_df['Data_Type'] == data_type]
        sns.scatterplot(data=subset, x=x_title, y=y_title, hue='Combination', style='Combination', ax=ax)
        ax.set_title(data_type)
        ax.legend().remove()  # We will create a single legend later
    
    # Remove any empty subplots
    for j in range(i + 1, n_rows * 2):
        fig.delaxes(axes[j // 2][j % 2])

    # Create a single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    plt.tight_layout(pad=5.0)  # Adjust the layout to make space for the legend
    plt.show()

    # Creating DataFrames for 'best_model_train' and 'best_model_test'
    for data_type in ['best_model_train', 'best_model_test']:
        subset = metrics_df[metrics_df['Data_Type'] == data_type][[y_title, x_title, 'Data', 'Model', 'Label', 'Weight']]
        subset = subset.sort_values(by=[y_title, x_title], ascending=False)
        print(f"DataFrame for {data_type}:\n{tabulate(subset, headers='keys', tablefmt='psql', showindex=False)}")


def plot_weights_distribution(weights: pd.Series):
    """
    Plots the distribution of weight values.

    :param weights: A pandas Series containing weight values.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(weights, kde=True, binwidth=0.05)#, binrange=(0, 1))
    plt.title('Distribution of Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    # plt.xlim(0, 1)  # Ensure x-axis is from 0 to 1
    plt.grid(True)
    plt.show()


def plot_returns_distribution(returns_s):
    """
    Plots the distribution of interval returns.

    :param returns_s: A pandas Series containing interval returns.
    """
    if not isinstance(returns_s, pd.Series):
        raise TypeError("returns_s must be a pandas Series")

    plt.figure(figsize=(10, 6))
    plt.hist(returns_s.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    plt.title('Distribution of Interval Returns')
    plt.xlabel('Interval Return')
    plt.ylabel('Number of Samples')
    plt.grid(True)
    plt.show()


def aggregate_trend_counts(returns_s: pd.Series):
    """
    Aggregates the counts of each unique return value, accounting for trends.
    
    :param returns_s: A pandas Series containing interval returns.
    """
    trend_counts = {}
    trend_value = None
    trend_length = 0

    for value in returns_s.values:
        if value != trend_value:
            if trend_value is not None:
                if trend_value not in trend_counts:
                    trend_counts[trend_value] = {'sample_num': 0, 'trend_num': 0}
                trend_counts[trend_value]['sample_num'] += trend_length
                trend_counts[trend_value]['trend_num'] += 1
            trend_value = value
            trend_length = 1
        else:
            trend_length += 1

    # Add the last trend
    if trend_value is not None:
        if trend_value not in trend_counts:
            trend_counts[trend_value] = {'sample_num': 0, 'trend_num': 0}
        trend_counts[trend_value]['sample_num'] += trend_length
        trend_counts[trend_value]['trend_num'] += 1

    # Convert counts to average number of samples per trend
    trend_averages = {k: v['sample_num'] / v['trend_num'] for k, v in trend_counts.items()}
    return trend_averages

def plot_1_aggregated_trend_distribution(returns_s):
    """
    Plots the distribution of interval returns showing individual data points on the plot.

    :param returns_s: A pandas Series containing interval returns.
    """
    aggregated_counts = aggregate_trend_counts(returns_s.dropna())
    
    # Prepare the data for plotting
    keys = list(aggregated_counts.keys())
    values = list(aggregated_counts.values())

    plt.figure(figsize=(10, 6))
    
    # Plot individual data points
    plt.scatter(keys, values, color='navy')

    plt.title('Aggregated Trend Distribution of Interval Returns')
    plt.xlabel('Interval Return')
    plt.ylabel('Average Number of Samples per Trend')
    plt.grid(True)
    plt.show()


def plot_2_aggregated_trend_distribution(returns_s, bins=50):
    """
    Plots the distribution of interval returns with aggregated trend counts.
    This time, returns are binned and the average count per bin is plotted.

    :param returns_s: A pandas Series containing interval returns.
    :param bins: The number of bins to use for the histogram.
    """
    aggregated_counts = aggregate_trend_counts(returns_s.dropna())
    # Convert the dictionary to a Series for easier manipulation
    aggregated_counts_series = pd.Series(aggregated_counts)

    # Define the bin edges
    bin_edges = np.linspace(returns_s.min(), returns_s.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Calculate the counts per bin
    counts_per_bin = np.zeros_like(bin_centers)
    for i in range(len(bin_edges) - 1):
        # Find the keys that fall into the current bin range
        keys_in_bin = aggregated_counts_series.index[(aggregated_counts_series.index >= bin_edges[i]) &
                                                    (aggregated_counts_series.index < bin_edges[i + 1])]
        # Calculate the average count for the keys in the current bin
        if len(keys_in_bin) > 0:
            counts_per_bin[i] = aggregated_counts_series[keys_in_bin].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts_per_bin, width=np.diff(bin_edges), align='center', color='blue', edgecolor='black')

    plt.title('Aggregated Trend Distribution of Interval Returns')
    plt.xlabel('Interval Return')
    plt.ylabel('Average Count per Bin')
    plt.grid(True)
    plt.show()


def aggregate_trend(returns_s: pd.Series):
    """
    Aggregates individual trend counts with their return value.

    :param returns_s: A pandas Series containing interval returns.
    :returns: A list of tuples, each containing the return value of the trend and its sample number.
    """
    trends = []
    trend_value = returns_s.iloc[0]
    trend_count = 1  # Start with the first sample

    for value in returns_s.iloc[1:]:
        if value == trend_value:
            # Continue the trend
            trend_count += 1
        else:
            # End of the trend, record it
            trends.append((trend_value, trend_count))
            # Start a new trend
            trend_value = value
            trend_count = 1  # Reset the count for the new trend

    # Catch the last trend if it goes till the end of the series
    trends.append((trend_value, trend_count))

    return trends


def plot_3_aggregated_trend_distribution(returns_s):
    """
    Plots each trend with its return value and corresponding sample number.

    :param returns_s: A pandas Series containing interval returns.
    """
    trends = aggregate_trend(returns_s.dropna())

    # Unpack the trend data
    returns, samples = zip(*trends)

    plt.figure(figsize=(10, 6))
    
    # Plot individual trends
    plt.scatter(returns, samples, color='navy')

    plt.title('Individual Trend Distribution of Interval Returns')
    plt.xlabel('Interval Return')
    plt.ylabel('Number of Samples in Trend')
    plt.grid(True)
    plt.show()


def plot_4_aggregated_trend_distribution(title, returns_s):
    """
    Plots each trend with its return value and corresponding sample number
    along with a linear regression line.
    
    :param returns_s: A pandas Series containing interval returns.
    """
    trends = aggregate_trend(returns_s.dropna())

    # Unpack the trend data
    returns, samples = zip(*trends)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(returns, samples)

    # Create a range of x values for the line of best fit
    x = np.array(returns)
    y = slope * x + intercept

    plt.figure(figsize=(10, 6))
    
    # Plot individual trends as scatter points
    plt.scatter(returns, samples, color='navy', label='Trend Data')

    # Plot the line of best fit
    plt.plot(x, y, color='red', label=f'Linear Regression (r^2 = {r_value ** 2:.2f})')

    plt.title(title)
    plt.xlabel('Weight value')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_highlighted_LW_table(metrics_dict, metric_name, metric_key, data_type):
    def parse_combination_name(name):
        parts = name.split(';')
        return {p.split('-')[0]: p.split('-')[1] for p in parts}

    def flatten_data():
        flattened_data = []
        for combination, data in metrics_dict.items():
            parsed_name = parse_combination_name(combination)
            for _data_type, metrics in data.items():
                if _data_type != data_type:
                    continue
                row = [combination]
                row.extend([parsed_name['D'], parsed_name['M'], parsed_name['L'], parsed_name['W'], _data_type])
                row.extend([metrics[metric_key]])
                flattened_data.append(row)
        return flattened_data

    flattened_data = flatten_data()
    columns = ['Combination', 'Data', 'Model', 'Label', 'Weight', 'Data_Type', metric_name]
    metrics_df = pd.DataFrame(flattened_data, columns=columns)
    # Pivot the DataFrame
    pivot_df = metrics_df.pivot(index='Label', columns='Weight', values=metric_name)

    highlighted_df = pivot_df.copy()

    # Function to add a marker to the maximum values
    def mark_max(row):
        max_val = row.max()
        return row.apply(lambda x: f"**{x}**" if x == max_val else x)

    # Apply the function across the DataFrame rows
    highlighted_df = highlighted_df.apply(mark_max, axis=1)

    none_columns = [col for col in pivot_df.columns if 'none' in col]
    trend_interval_columns = [col for col in pivot_df.columns if 'trend_interval' in col]
    other_columns = [col for col in pivot_df.columns if col not in none_columns + trend_interval_columns]
    new_order = none_columns + trend_interval_columns + other_columns
    reordered_df = highlighted_df[new_order]

    print(f"Table for {metric_name}:")
    print(tabulate(reordered_df, headers='keys', tablefmt='psql', showindex=True))
    print()
