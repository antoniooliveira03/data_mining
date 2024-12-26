# This file will consist of functions used through the notebooks

#################### Imports ##############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.base import clone
from sklearn.cluster import KMeans


#################### Histograms ##############################

def draw_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 6))  
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=10, ax=ax, color='#4CAF50', edgecolor='black') 
        ax.set_title(var_name + " Distribution")
        ax.grid(False)
    fig.tight_layout()
    plt.show()

#################### Missing values summary ##############################
def missing_value_summary(dataframe):
    nan_columns = dataframe.columns[dataframe.isna().any()].tolist()
    
    summary_data = []
    
    for column in nan_columns:

        nan_number = dataframe[column].isna().sum()

        nan_percentage = (nan_number / len(dataframe)) * 100

        unique_values = dataframe[column].nunique()
        
        summary_data.append({
            'Unique Values': unique_values,
            'NaN Values': nan_number,
            'Percentage NaN': nan_percentage
        })
    
    summary = pd.DataFrame(summary_data, index=nan_columns)
    
    return summary

# Missing Values
# Create a function to impute missing values of age
def impute_age(row):
    if np.isnan(row['customer_age']):
        return int(round(age_medians.get(row['region_cuisine_group'], np.nan)))
    else:
        return row['customer_age']
    
#################### Outliers ###############################
def IQR_outliers(df: pd.DataFrame,
                  variables: list[str]
                  ) -> None:
    """
    Identify potential outliers using the interquartile
      range (IQR) method.

    ----------
    Parameters:
     - df (pd.DataFrame): The pandas dataframe to be
        analyzed.
     - variables (list): A list of column names in the
        dataframe to check for outliers.

    ----------
    Returns:
     - None, but prints the potential outliers for each
        variable along with the number of outliers.
    """

    # Calculate the IQR for each variable
    q1 = df[variables].quantile(0.25)
    q3 = df[variables].quantile(0.75)
    iqr = q3 - q1

    # Identify potential outliers for each variable
    lower_bound = q1 - (3 * iqr)
    upper_bound = q3 + (3 * iqr)

    outliers = {}
    for var in variables:
        outliers[var] = df[(df[var] < lower_bound[var]) | (df[var] > upper_bound[var])][var]

    # Print the potential outliers for each variable
    print('-------------------------------------')
    print('          Potential Outliers         ')
    print('-------------------------------------')

    for var in outliers:
        print(var, ': Number of Outliers ->', len(outliers[var]))
        if len(outliers[var]) != 0:
            outliers[var] = np.unique(outliers[var])
            print('  Outliers: ',outliers[var])
        print()


def plot_multiple_boxes_with_outliers1(data, columns, ncols=2):
    """
    Plots box plots for specified columns in the DataFrame and highlights the outliers.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names to plot.
    ncols (int): Number of columns in the subplot grid.
    """
    num_columns = len(columns)
    nrows = (num_columns + ncols - 1) // ncols  # Calculates the number of rows needed

    plt.figure(figsize=(8 * ncols, 4 * nrows))

    for i, column in enumerate(columns):
        # Calculate quartiles and IQR
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Determine the outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]

        # Create the box plot for the current column
        plt.subplot(nrows, ncols, i + 1)  # Create a subplot grid
        plt.boxplot(data[column], vert=False, widths=0.7,
                    patch_artist=True, boxprops=dict(facecolor='#4CAF50', color='black'),
                    medianprops=dict(color='black'))

        # Scatter outliers
        plt.scatter(outliers, [1] * len(outliers), color='red', marker='o', label='Outliers')

        # Customize the plot
        plt.title(f'Box Plot of {column} with Outliers')
        plt.xlabel('Value')
        plt.yticks([])
        plt.legend()

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()


def cap_outliers(data):
    
    for column in data.columns:
        # Calculating the quartiles and interquartile range
        q1 = np.percentile(data[column], 25)
        q3 = np.percentile(data[column], 75)
        iqr = q3 - q1

        # Setting the boundaries for outliers
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Capping the outliers
        data[column] = data[column].apply(
            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
        )

#################### Feature Engineering ##############################
def avg_hour(row):
    """
    Computes the average hour when orders were placed, 
    weighted by the number of orders at each hour.
    """
    total_orders = row.sum()
    
    if total_orders == 0:
        return None  
    
    weighted_sum_hours = (row.index.str.replace('HR_', '').astype(int) * row).sum()
    return weighted_sum_hours / total_orders

#################### Clustering ##############################

def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    
    # Plot the dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_dim_reduction(embedding, targets = None, 
                       technique = 'UMAP',
                       figsize = (10, 7)):
    
    plt.figure(figsize=figsize)
    
    if targets is not None:
        # Ensure targets are in integer format for color mapping
        labels = np.unique(targets)
        scatter = plt.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=np.array(targets).astype(int), 
            cmap='tab10'
        )

        # Create a legend with the class labels and colors
        handles = [plt.scatter([], [], color=plt.cm.tab10(i), label=label) for i, label in enumerate(labels)]
        plt.legend(handles=handles, title='Clusters')

        
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5)

    if technique == 'UMAP':
        plt.title('UMAP Projection')
    elif technique == 't-SNE':
        plt.title('t-SNE Projection')
    else:
        plt.title(f'{technique} Projection')

    plt.show()


def calculate_r2(df:pd.DataFrame,
                feats:list,
                label_col:str) -> float:
    """
    Calculate the R-squared value for a given DataFrame and features.

    ------------------------------------------  
    Arguments:
     - df (pd.DataFrame): The input DataFrame.
     - feats (list): List of feature column names.
     - label_col (str): Name of the column containing labels.
        
    ------------------------------------------
    Returns:
     - float: The R-squared value.
    """
    overall_mean = df[feats].mean()
    group_means = df.groupby(label_col)[feats].mean()
    group_sizes = df.groupby(label_col)[feats].count()
    ssb = np.sum(group_sizes * np.square(group_means - overall_mean).sum(axis=1))
    sst = np.sum(df[feats].var() * (df[feats].count() - 1))
    return ssb / sst


def clust_diff_k(df: pd.DataFrame, 
                feats: list, 
                clusterer: KMeans, 
                min_k: int = 2, 
                max_k: int = 10) -> dict:
    """
    Loop over different values of k. To be used with sklearn clusters.

    ------------------------------------------
    Arguments:
     - df: The input DataFrame.
     - feats: List of feature column names.
     - clusterer: The sklearn clustering model (e.g., KMeans).
     - min_k: Minimum number of clusters. Defaults to 2.
     - max_k: Maximum number of clusters. Defaults to 10.

    ------------------------------------------
    Returns:
    - dict: A dictionary where keys are the number of clusters (k) 
        and values are the corresponding R-squared scores.
    """
    r2_clust = {}
    for n in range(min_k, max_k):
        clust = clone(clusterer).set_params(n_clusters=n)
        labels = clust.fit_predict(df)
        df_concat = pd.concat([df, 
                              pd.Series(labels, name='labels', index=df.index)], axis=1) 
        r2_clust[n] = calculate_r2(df_concat, feats, 'labels')
    return r2_clust