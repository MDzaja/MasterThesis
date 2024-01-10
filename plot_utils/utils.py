import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.ticker import MultipleLocator


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
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
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
