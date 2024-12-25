import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a default style for plots (optional):
sns.set_style("whitegrid")

def eda_summary_stats(df):
    """
    Prints a summary of the DataFrame dimensions and basic statistical information 
    for both numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.

    Returns
    -------
    None
    """
    print("=== DATAFRAME DIMENSIONS ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\n=== STATISTICAL DESCRIPTION (NUMERIC COLUMNS) ===")
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        display(df.describe(include=[np.number]))
    else:
        print("No numeric columns found.")
    
    print("\n=== STATISTICAL DESCRIPTION (CATEGORICAL COLUMNS) ===")
    if df.select_dtypes(include=['object', 'category']).shape[1] > 0:
        display(df.describe(include=['object', 'category']))
    else:
        print("No categorical columns found.")


def eda_missing_values(df):
    """
    Returns the count and percentage of missing values for each column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the columns, count of missing values, and percentage of missing values.
    """
    missing_count = df.isnull().sum()
    missing_percentage = 100 * (missing_count / len(df))

    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing': missing_count,
        'percent': missing_percentage
    }).reset_index(drop=True)
    
    return missing_df


def eda_plot_missing_values(df, figsize=(8, 6)):
    """
    Plots a bar chart showing the number of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    figsize : tuple, default=(8,6)
        Figure size for the plot.

    Returns
    -------
    None
    """
    missing_df = eda_missing_values(df)
    missing_df = missing_df[missing_df['missing'] > 0].sort_values(
        by='missing', ascending=False
    )

    if missing_df.empty:
        print("There are no missing values in the DataFrame.")
        return

    plt.figure(figsize=figsize)
    sns.barplot(
        x='missing',
        y='column',
        data=missing_df,
        palette='viridis'
    )
    plt.title('Missing Values by Column')
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Columns')
    plt.show()


def eda_correlation_heatmap(df, method='pearson', annot_fontsize=8):
    """
    Plots a heatmap showing the correlation matrix for numeric variables in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', or 'kendall').
    annot_fontsize : int, default=8
        Font size for the annotation text in the heatmap.

    Returns
    -------
    None
    """
    corr_matrix = df.corr(method=method)
    num_cols = len(corr_matrix.columns)
    figsize = (num_cols * 0.5 + 5, num_cols * 0.5 + 5)  # Adjust size dynamically
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        annot_kws={"size": annot_fontsize},  # Set font size for annotations
        cbar_kws={'shrink': 0.8}  # Adjust color bar size for readability
    )
    plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Automatically adjust layout
    plt.show()


def eda_distribution_plots(df, cols=None, bins=30, hist_kws=None):
    """
    Plots histograms and density curves (kde) for numeric columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    cols : list or None
        List of specific columns to plot. If None, all numeric columns will be plotted.
    bins : int, default=30
        Number of bins for the histogram.
    hist_kws : dict or None
        Additional parameters for the seaborn histogram (histplot).

    Returns
    -------
    None
    """
    if hist_kws is None:
        hist_kws = {}

    # If no columns are specified, use all numeric columns
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 4))
            sns.histplot(data=df, x=col, kde=True, bins=bins, **hist_kws)
            plt.title(f'Distribution of {col}')
            plt.show()


def eda_line_plot_sampled_smoothed(df, x_col, y_col, sample_size=10000, window=50, figsize=(10, 6)):
    """
    Plots a smoothed line showing the trend of a numeric column (y_col) 
    with respect to another column (x_col) using a sample of the data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    x_col : str
        The name of the column for the x-axis.
    y_col : str
        The name of the numeric column for the y-axis.
    sample_size : int, default=10000
        The number of rows to sample for faster plotting.
    window : int, default=50
        The window size for calculating the moving average.
    figsize : tuple, default=(10, 6)
        The size of the figure.

    Returns
    -------
    None
    """
    # Muestreo si el número de filas excede el tamaño de la muestra
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # Ordenar por x_col para asegurar una línea continua
    df = df.sort_values(by=x_col)

    # Calcular promedio móvil
    df[y_col + '_smoothed'] = df[y_col].rolling(window=window, center=True).mean()

    plt.figure(figsize=figsize)
    sns.lineplot(data=df, x=x_col, y=y_col + '_smoothed', color='blue', label='Trend')
    plt.title(f'Smoothed Trend of {y_col} with respect to {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def eda_class_continuous_plot(df, class_col, continuous_col, plot_type='box', figsize=(10, 6)):
    """
    Plots the relationship between a binary class column and a continuous variable 
    using a boxplot or violin plot.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    class_col : str
        The name of the binary class column (e.g., 0 or 1).
    continuous_col : str
        The name of the continuous variable column.
    plot_type : str, default='box'
        The type of plot ('box' for boxplot, 'violin' for violin plot).
    figsize : tuple, default=(10, 6)
        The size of the figure.

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    
    if plot_type == 'box':
        sns.boxplot(data=df, x=class_col, y=continuous_col)
        plt.title(f'Boxplot of {continuous_col} by {class_col}')
    elif plot_type == 'violin':
        sns.violinplot(data=df, x=class_col, y=continuous_col)
        plt.title(f'Violin Plot of {continuous_col} by {class_col}')
    else:
        raise ValueError("plot_type must be 'box' or 'violin'")
    
    plt.xlabel(class_col)
    plt.ylabel(continuous_col)
    plt.tight_layout()
    plt.show()


def eda_line_plot_sampled(df, x_col, y_col, sample_size=10000, figsize=(10, 6)):
    """
    Plots the evolution of a numeric column (y_col) with respect to another column (x_col)
    using a random sample of the DataFrame for faster plotting.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    x_col : str
        The name of the column for the x-axis.
    y_col : str
        The name of the numeric column for the y-axis.
    sample_size : int, default=10000
        The number of random rows to sample for the plot.
    figsize : tuple, default=(10, 6)
        The size of the figure.

    Returns
    -------
    None
    """
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    plt.figure(figsize=figsize)
    sns.lineplot(data=df, x=x_col, y=y_col, marker='o', alpha=0.7)
    plt.title(f'Evolution of {y_col} with respect to {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def eda_boxplot_for_numerical(df, cols=None):
    """
    Plots boxplots for numeric columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    cols : list or None
        List of specific numeric columns to plot. If None, all numeric columns will be plotted.

    Returns
    -------
    None
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns

    for col in cols:
        plt.figure(figsize=(4, 6))
        sns.boxplot(y=df[col], color='lightblue')
        plt.title(f'Boxplot of {col}')
        plt.show()


def eda_countplot_for_categorical(df, cols=None):
    """
    Plots countplots for categorical columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    cols : list or None
        List of specific categorical columns to plot. If None, all categorical columns will be plotted.

    Returns
    -------
    None
    """
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[col], palette='viridis')
        plt.title(f'Count Plot for {col}')
        plt.xticks(rotation=45)
        plt.show()


def eda_pairplot(df, cols=None, hue=None):
    """
    Creates a pairplot (scatter plot matrix) for a subset of numeric columns, optionally 
    colored by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    cols : list or None
        List of specific numeric columns to include in the pairplot. If None, all numeric columns are used.
    hue : str or None
        Column name for categorization (coloring the plots).

    Returns
    -------
    None
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns

    sns.pairplot(df[cols], hue=hue, diag_kind='kde', corner=True)
    plt.show()


def eda_outliers_iqr(df, col):
    """
    Detects outliers in a specific column using the Interquartile Range (IQR) rule.
    Returns a DataFrame containing the outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    col : str
        The name of the numeric column to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the outliers found in the specified column.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers_df


def eda_countplot_categorical_percentages(df, col):
    """
    Plotea un countplot para una columna categórica y muestra
    el porcentaje sobre cada barra.
    """
    # Calculamos frecuencias absolutas y porcentajes
    counts = df[col].value_counts()
    total = counts.sum()
    percentages = counts / total * 100

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=counts.index, y=counts.values)

    # Anotar valores en cada barra
    for i, val in enumerate(counts.values):
        pct = f"{percentages.values[i]:.1f}%"
        ax.text(i, val, pct, ha='center', va='bottom', fontsize=10, color='black')

    plt.title(f'Count Plot y Porcentajes para {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def eda_top_n_categorical(df, col, n=5):
    """
    Returns the top-n most frequent values of a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    col : str
        The name of the categorical column.
    n : int, default=5
        The number of top frequent values to return.

    Returns
    -------
    pd.Series
        A Series containing the most frequent values and their counts.
    """
    return df[col].value_counts().head(n)


def eda_groupby_stats(df, group_col, agg_col, agg_func='mean'):
    """
    Groups the DataFrame by a categorical column and computes an aggregated metric 
    (e.g., mean) on another numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    group_col : str
        The name of the categorical column to group by.
    agg_col : str
        The name of the numeric column to aggregate.
    agg_func : str, default='mean'
        The aggregation function to apply (e.g., 'mean', 'sum', 'max', etc.).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the grouped categorical column and the aggregated metric.
    """
    grouped_df = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    grouped_df.columns = [group_col, f'{agg_func}_{agg_col}']
    return grouped_df


def eda_plot_grouped_bar(df, group_col, agg_col, agg_func='mean', figsize=(8,6)):
    """
    Creates a bar plot from a grouped DataFrame, using a categorical column for grouping 
    and computing an aggregated metric on another numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed.
    group_col : str
        The name of the categorical column to group by.
    agg_col : str
        The name of the numeric column to aggregate.
    agg_func : str, default='mean'
        The aggregation function to apply for the numeric column.
    figsize : tuple, default=(8,6)
        The size of the figure.

    Returns
    -------
    None
    """
    grouped_df = eda_groupby_stats(df, group_col, agg_col, agg_func)
    
    plt.figure(figsize=figsize)
    sns.barplot(
        x=group_col, 
        y=f'{agg_func}_{agg_col}', 
        data=grouped_df, 
        palette='viridis'
    )
    plt.title(f'{agg_func.capitalize()} of {agg_col} grouped by {group_col}')
    plt.xticks(rotation=45)
    plt.show()