## Chat Reference: 
# - https://chatgpt.com/c/6705c501-d5c0-8006-bd53-1c7f752864cc

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_histogram(results_df, metric, axis, angle):
    """
    Plots a histogram of the specified metric at a given rotation axis and angle.
    Args:
        results_df (pd.DataFrame): The evaluation results DataFrame.
        metric (str): The evaluation metric to plot ('RMSE', 'MAE', etc.).
        axis (str): Rotation axis ('roll', 'pitch', 'yaw').
        angle (float): Rotation angle.
    """
    subset = results_df[(results_df['rotation_axis'] == axis) & (results_df['rotation_angle'] == angle)]
    plt.figure(figsize=(10, 6))
    sns.histplot(subset[metric], bins=20, kde=True)
    plt.title(f'Histogram of {metric} at {axis} = {angle}°')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.show()

def plot_metric_statistics(results_df, metric, axis):
    """
    Plots mean, variance, min, and max of the specified metric across rotation angles for a given axis.
    Args:
        results_df (pd.DataFrame): The evaluation results DataFrame.
        metric (str): The evaluation metric to analyze ('RMSE', 'MAE', etc.).
        axis (str): Rotation axis ('roll', 'pitch', 'yaw').
    """
    subset = results_df[results_df['rotation_axis'] == axis]
    stats_df = subset.groupby('rotation_angle')[metric].agg(['mean', 'var', 'min', 'max']).reset_index()
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(stats_df['rotation_angle'], stats_df['mean'], yerr=stats_df['var']**0.5, fmt='-o', label='Mean ± Std')
    plt.fill_between(stats_df['rotation_angle'], stats_df['min'], stats_df['max'], color='gray', alpha=0.2, label='Min/Max Range')
    plt.title(f'{metric} Statistics across {axis} Rotation')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load the evaluation results
    results_df = pd.read_csv('evaluation_results.csv')
    
    # Example 1: Plot histogram of RMSE at roll = 0.0°
    plot_metric_histogram(results_df, metric='RMSE', axis='roll', angle=0.0)
    
    # Example 2: Plot RMSE statistics across pitch rotation
    plot_metric_statistics(results_df, metric='RMSE', axis='pitch')