import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import itertools
from functools import partial
import concurrent.futures
from tqdm import tqdm
import threading

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
    ax2.plot(union_index, labels_, color='orange', label='Trend labels', linestyle='-', marker='.')
    ax2.set_ylabel('Label', color='black')
    ax2.yaxis.set_major_locator(MultipleLocator(1))

    # Show the plot
    plt.show()

# Global progress bar and lock
pbar = None
lock = threading.Lock()

def optimize_label_params(binary_trend_labels: callable, prices: pd.Series, param_grid: list[list], fee: float=0, num_threads: int=1) -> tuple:
    max_return = float('-inf')
    best_params = None
    
    params_list = list(itertools.product(*param_grid))

    # Split parameter combinations amongst threads
    chunks = [params_list[i::num_threads] for i in range(num_threads)]

    print(f"Number of parameter combinations: {len(params_list)}")
    print(f"Number of threads: {num_threads}, Number of parameter combinations per thread: {len(chunks[0])}")

    # Initialize the progress bar
    global pbar
    pbar = tqdm(total=len(params_list), desc="Optimizing", ncols=100)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(worker, 
                                    [binary_trend_labels]*num_threads,
                                    [prices]*num_threads,
                                    chunks,
                                    [fee]*num_threads))
        
        for current_return, params in results:
            if current_return > max_return:
                max_return = current_return
                best_params = params

    pbar.close()  # Close the progress bar
    return best_params

def worker(binary_trend_labels: callable, prices: pd.Series, params_subset: list[tuple], fee: float) -> tuple:
    max_return_thread = float('-inf')
    best_params_thread = None

    for params in params_subset:
        partial_func = partial(binary_trend_labels, prices)
        labels = partial_func(*params)
        current_return = compute_return(prices, labels, fee)

        if current_return > max_return_thread:
            max_return_thread = current_return
            best_params_thread = params
        
        # Update the progress bar
        with lock:
            pbar.update(1)
            
    return (max_return_thread, best_params_thread)

def compute_return(prices: pd.Series, labels: pd.Series, fee: float=0) -> float:
    if len(prices) != len(labels):
        raise ValueError("The length of prices and labels must be the same.")

    cumulative_return = 1.0
    i = 0

    while i < len(labels):
        if labels.iloc[i] == 1:  # Start of a long position
            start_price = prices.iloc[i]
            while i < len(labels) and labels.iloc[i] == 1:
                i += 1
            end_price = prices.iloc[i-1] if i > 0 else start_price
            end_price *= (1 - fee)
            cumulative_return *= (end_price / start_price)
        else:
            i += 1

    return cumulative_return - 1.0  # Subtract 1 to get the actual return, not the factor
    