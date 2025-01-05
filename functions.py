# This file will consist of functions used through the notebooks

## IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
from sklearn.base import clone
import matplotlib.cm as cm
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.colors as mpl_colors
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colorbar

# Define the main color
main_color = '#568789'

## FUNCTIONS

### HISTOGRAM

def draw_histograms(df, variables, n_rows, n_cols):
    """
    Draws histograms for the specified variables in a grid layout.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        variables (list): List of variables to plot.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
    """
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 6))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=10, ax=ax, color='#4CAF50', edgecolor='black')
        ax.set_title(f"{var_name} Distribution")
        ax.grid(False)
    fig.tight_layout()
    plt.show()

### MISSING VALUES SUMMARY

def missing_value_summary(dataframe):
    """
    Provides a summary of missing values in the DataFrame.
    
    Parameters:
        dataframe (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
        pd.DataFrame: Summary of columns with missing values, including unique values, NaN count, and percentage.
    """
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

### MISSING VALUES IMPUTATION - age

def impute_age(row):
    """
    Imputes missing age values based on median age grouped by region and cuisine.
    
    Parameters:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        int: Imputed or original age value.
    """
    if np.isnan(row['customer_age']):
        return int(round(age_medians.get(row['region_cuisine_group'], np.nan)))
    return row['customer_age']

### OUTLIER DETECTION

def IQR_outliers(df, variables):
    """
    Identifies potential outliers using the interquartile range (IQR) method.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        variables (list): List of columns to check for outliers.
    """
    q1 = df[variables].quantile(0.25)
    q3 = df[variables].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (3 * iqr)
    upper_bound = q3 + (3 * iqr)

    outliers = {}
    for var in variables:
        outliers[var] = df[(df[var] < lower_bound[var]) | (df[var] > upper_bound[var])][var]

    print('-------------------------------------')
    print('          Potential Outliers         ')
    print('-------------------------------------')

    for var in outliers:
        print(f"{var} : Number of Outliers -> {len(outliers[var])}")
        if len(outliers[var]) != 0:
            outliers[var] = np.unique(outliers[var])
            print(f"  Outliers: {outliers[var]}")
        print()

### BOX PLOT WITH OUTLIERS

def plot_multiple_boxes_with_outliers(data, columns, ncols=2):
    """
    Plots box plots for specified columns and highlights the outliers.
    
    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to plot.
        ncols (int): Number of columns in the subplot grid.
    """
    num_columns = len(columns)
    nrows = (num_columns + ncols - 1) // ncols
    plt.figure(figsize=(8 * ncols, 4 * nrows))

    for i, column in enumerate(columns):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]

        plt.subplot(nrows, ncols, i + 1)
        plt.boxplot(data[column], vert=False, widths=0.7,
                    patch_artist=True, boxprops=dict(facecolor='#4CAF50', color='black'),
                    medianprops=dict(color='black'))
        plt.scatter(outliers, [1] * len(outliers), color='red', marker='o', label='Outliers')
        plt.title(f"Box Plot of {column} with Outliers")
        plt.xlabel('Value')
        plt.yticks([])
        plt.legend()

    plt.tight_layout()
    plt.show()

### OUTLIER CAPPING

def cap_outliers(data):
    """
    Caps outliers in the DataFrame columns using the IQR method.
    
    Parameters:
        data (pd.DataFrame): The DataFrame to modify.
    """
    for column in data.columns:
        q1 = np.percentile(data[column], 25)
        q3 = np.percentile(data[column], 75)
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        data[column] = data[column].apply(
            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
        )

### DISTRIBUTION AND BOX PLOT

def plot_distribution_and_boxplot(df, column_name, color=main_color):
    """
    Plots the distribution and box plot for a specific column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): Column to visualize.
        color (str): Plot color.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(df[column_name], kde=True, bins=30, color=color, ax=axes[0])
    axes[0].set_title(f"Distribution of {column_name}")
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=df[column_name], color=color, ax=axes[1])
    axes[1].set_title(f"Boxplot of {column_name}")
    axes[1].set_xlabel(column_name)

    plt.tight_layout()
    plt.show()

### FEATURE ENGINEERING

def avg_hour(row):
    """
    Computes the average hour when orders were placed, weighted by the number of orders.
    
    Parameters:
        row (pd.Series): Row containing hour columns.
    
    Returns:
        float: Weighted average hour or 0 if no orders were placed.
    """
    total_orders = row.sum()
    if total_orders == 0:
        return 0
    
    weighted_sum_hours = (row.index.str.replace('HR_', '').astype(int) * row).sum()
    return weighted_sum_hours / total_orders

### CLUSTERING

# Plot Distribution 
def plot_counts(labels):
    """
    Plots a bar chart showing the counts of each cluster label.

    Parameters:
    - labels (array-like): Cluster labels for data points.

    """
    label_counts = pd.Series(labels).value_counts()
    plt.figure(figsize=(8, 6))
    label_counts.plot(kind='bar', color=main_color)
    plt.title('Cluster Label Counts')
    plt.xlabel('Cluster Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Plot Hierarchical Clustering Dendrogram

def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
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

def plot_hierarchical_dendrograms(data, path=None, linkages=["ward", "complete", "average", "single"], metrics=['euclidean']):
    """
    Create and display a grid of hierarchical clustering dendrograms for the given data,
    enumerating different linkage and metric combinations.

    Parameters:
    - data: The dataset to cluster, should be a NumPy array or a pandas DataFrame.
    - path: defines where the image will be saved.
    - linkages: List of linkage methods to evaluate (default is ['ward', 'complete', 'average', 'single']).
    - metrics: List of distance metrics to use for the AgglomerativeClustering (default is ['euclidean']).
    """
    # Number of subplots we need for combinations of linkages and metrics
    num_plots = len(linkages) * len(metrics)

    # Create a plot grid based on the number of combinations
    num_rows = (num_plots + 1) // 2  # Adjust grid size to fit all combinations
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))
    axes = axes.ravel()  # Flatten the axes array for easy access

    # Loop through both linkage and metric combinations
    plot_idx = 0
    for linkage in linkages:
        for metric in metrics:
            # Perform AgglomerativeClustering with the current linkage and metric
            model = AgglomerativeClustering(
                linkage=linkage, distance_threshold=0, n_clusters=None, metric=metric
            ).fit(data)
            
            # Plot dendrogram on the corresponding subplot
            ax = axes[plot_idx]  # Get the corresponding axis
            ax.set_title(f"Linkage: {linkage} - Metric: {metric}")
            
            # Plot the dendrogram
            plot_dendrogram(model, ax=ax, truncate_mode="level", p=10)

            # Increment the subplot index
            plot_idx += 1

    # Adjust layout for better visibility
    plt.tight_layout()
    if path != None:
        plt.savefig(f'{path}_dendrogram.png')
    plt.show()


# Plot T-SNE or UMAP visualisation
def plot_dim_reduction(embedding, targets=None, 
                       technique='UMAP',
                       figsize=(10, 7)):

    """
    Plots a 2D representation of high-dimensional data.

    Parameters:
    - embedding (array-like): 2D array of transformed data.
    - targets (array-like): Cluster labels for data points (optional).
    - technique (str): Dimensionality reduction technique name (default: 'UMAP').
    - figsize (tuple): Figure size (default: (10, 7)).
    """
    plt.figure(figsize=figsize)

    if targets is not None:
        # Ensure targets are in integer format for color mapping
        scatter = plt.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=np.array(targets).astype(int), 
            cmap='tab10'
        )

        
        # Create a legend with the class labels and corresponding colors from the scatter plot
        labels = np.unique(targets)
        handles = []
        
        # Manually create handles using the same colormap as scatter
        for i, label in enumerate(labels):
            color = scatter.cmap(scatter.norm(i))  
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label))

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

# Plot SOM Hexagons
def plot_som_hexagons(som, matrix, cmap=cm.Blues, figsize=(20, 20),
                      annotate=True, title="SOM Matrix", cbar_label="Color Scale"):
    """
    Plots a hexagonal grid visualization for a Self-Organizing Map (SOM).

    Parameters:
    - som: A SOM object with `get_euclidean_coordinates()` method.
    - matrix (2D array): Matrix of values to display on the SOM grid.
    - cmap (matplotlib colormap): Colormap for the hexagons (default: cm.Blues).
    - figsize (tuple): Size of the plot figure (default: (20, 20)).
    - annotate (bool): Whether to annotate hexagons with matrix values (default: True).
    - title (str): Title of the plot (default: "SOM Matrix").
    - cbar_label (str): Label for the colorbar (default: "Color Scale").

    """

    xx, yy = som.get_euclidean_coordinates()

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=20)

    colornorm = mpl_colors.Normalize(vmin=np.min(matrix), 
                                     vmax=np.max(matrix))

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hexagon = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius=.95 / np.sqrt(3),
                                 facecolor=cmap(colornorm(matrix[i, j])), 
                                 alpha=1)
            ax.add_patch(hexagon)

            if annotate:
                annot_vals = np.round(matrix[i, j],2)
                if annot_vals > 1:
                    annot_vals = int(annot_vals)
                
                ax.text(xx[(i, j)], wy, annot_vals, 
                        ha='center', va='center', 
                        fontsize=figsize[1], 
                        )

    ax.margins(.05)
    ax.axis("off")

    ## Create a Mappable object
    cmap_sm = plt.cm.ScalarMappable(cmap=cmap, norm=colornorm)
    cmap_sm.set_array([])
    
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="2%", pad=0)    
    cb1 = colorbar.ColorbarBase(ax_cb, 
                                orientation='vertical', 
                                alpha=1,
                                mappable=cmap_sm
                               )
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel(cbar_label, fontsize=18)
    plt.gcf().add_axes(ax_cb)

    return plt

# Plots Cluster Profiling

def plot_cluster_profiling(df, cluster_labels, cluster_method_name, 
                           figsize=(6, 8), cmap="BrBG", fmt=".2f"):
    """
    Plots a heatmap showing the cluster profiling based on feature means.

    Args:
    - df (DataFrame): The original dataset with numerical features.
    - cluster_labels (array-like): Cluster labels for each data point.
    - cluster_method_name (str): Name of the clustering method (used in the title).
    - figsize (tuple): Size of the plot figure (default: (6, 8)).
    - cmap (str): Colormap for the heatmap (default: "BrBG").
    - fmt (str): String format for heatmap annotations (default: ".2f").
    """
    # Concatenate the cluster labels with the original data
    df_concat = pd.concat([df, pd.Series(cluster_labels, name='labels', index=df.index)], axis=1)
    
    # Group by cluster labels and compute the mean for each feature
    cluster_profile = df_concat.groupby('labels').mean().T
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    sns.heatmap(cluster_profile, center=0, annot=True, cmap=cmap, fmt=fmt, ax=ax)

    # Set labels and title
    ax.set_xlabel("Cluster Labels")
    ax.set_title(f"Cluster Profiling:\n{cluster_method_name} Clustering")
    
    # Show the plot
    plt.show()


# Sum of Squares
def get_ss(df, feats):
    """
    Calculate the sum of squares (SS) for the given DataFrame.

    The sum of squares is computed as the sum of the variances of each column
    multiplied by the number of non-NA/null observations minus one.

    Parameters:
    df (pandas.DataFrame): The input DataFrame for which the sum of squares is to be calculated.
    feats (list of str): A list of feature column names to be used in the calculation.

    Returns:
    float: The sum of squares of the DataFrame.
    """
    df_ = df[feats]
    ss = np.sum(df_.var() * (df_.count() - 1))
    
    return ss 


# Sum of Squares Between
def get_ssb(df, feats, label_col):
    """
    Calculate the sum of squares between (SSB) for the specified features and the label column.

    The sum of squares between is calculated by comparing the means of each group 
    (based on the label column) to the overall mean of the features.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to include in the calculation.
    label_col (str): The name of the label column used to group the data.

    Returns:
    float: The sum of squares between (SSB) for the specified features and label column.
    """
    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))) )

    ssb = np.sum(ssb_i)
    

    return ssb

# Sum of Squares Within (SSW)
def get_ssw(df, feats, label_col):
    """
    Calculate the sum of squares within (SSW) for the specified features and label column.

    The sum of squares within is computed by calculating the total sum of squares for 
    each group (based on the label column) and summing them across all groups.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to include in the calculation.
    label_col (str): The name of the label column used to group the data.

    Returns:
    float: The sum of squares within (SSW) for the specified features and label column.
    """
    feats_label = feats+[label_col]

    df_k = df[feats_label].groupby(by=label_col).apply(lambda col: get_ss(col, feats), 
                                                       include_groups=False)

    return df_k.sum()

# Residual Sum os Squares
def get_rsq(df, feats, label_col):
    """
    Calculate the R-squared (R²) value for the data, based on the total, within, and between 
    sum of squares.

    R² is computed as the ratio of the sum of squares between (SSB) to the total sum of squares (SST).

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to include in the calculation.
    label_col (str): The name of the label column used to group the data.

    Returns:
    float: The R-squared (R²) value.
    """
    df_sst_ = get_ss(df, feats)             
    df_ssw_ = get_ssw(df, feats, label_col) 
    df_ssb_ = df_sst_ - df_ssw_             

    # r2 = ssb/sst 
    return (df_ssb_/df_sst_)

# R-Squared
def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """
    Calculate the R-squared (R²) values for hierarchical clustering with different 
    numbers of clusters, using the specified linkage method and distance metric.

    The function computes the R² value for each clustering solution within the 
    specified range of clusters.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    link_method (str): The linkage method to use for hierarchical clustering (e.g., "ward", "single", "complete").
    max_nclus (int): The maximum number of clusters to consider.
    min_nclus (int, optional): The minimum number of clusters to consider (default is 1).
    dist (str, optional): The distance metric to use for clustering (default is "euclidean").

    Returns:
    numpy.ndarray: An array containing the R² values for each cluster solution.
    """
    r2 = []  
    feats = df.columns.tolist()
    
    for i in range(min_nclus, max_nclus+1):  
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)
        
        # Get cluster labels
        hclabels = cluster.fit_predict(df) 
        
        # Concat df with labels
        df_concat = pd.concat([df, pd.Series(hclabels, name='labels', index=df.index)], axis=1)  
        
        
        # Append the R2 of the given cluster solution
        r2.append(get_rsq(df_concat, feats, 'labels'))

        
    return np.array(r2)

# Cluster Quality Evaluation
def cluster_evaluation(df, feats, labels):
    """
    Evaluate the quality of a clustering solution using various metrics: R-squared (R²), 
    silhouette score, and Calinski-Harabasz score.

    The function computes the following metrics for the given clustering labels:
    - R-squared (R²): Measures the proportion of variance explained by the clusters.
    - Silhouette score: Evaluates the quality of the clusters based on both cohesion and separation.
    - Calinski-Harabasz score: Assesses the clustering quality by considering the variance between clusters and within clusters.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to include in the evaluation.
    labels (array-like): An array of cluster labels corresponding to each data point.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - R² values (array): The R-squared (R²) values for the clustering solution.
        - Silhouette scores (array): The silhouette scores for the clustering solution.
        - Calinski-Harabasz scores (array): The Calinski-Harabasz scores for the clustering solution.
    """
    r2 = []  
    silhouette = []
    calinski_harabasz = []

    # Concat df with labels
    df_concat = pd.concat([df, pd.Series(labels, name='labels', index=df.index)], axis=1)   

    # Append 
    r2.append(get_rsq(df_concat, feats, 'labels'))
    silhouette.append(silhouette_score(df, labels))
    calinski_harabasz.append(calinski_harabasz_score(df, labels))

    return np.array(r2), np.array(silhouette), np.array(calinski_harabasz)


# Create and Evaluate Models
def create_and_evaluate_model(df, feats, model_type, **kwargs):
    """
    Create a clustering model (KMeans, Hierarchical, etc.), fit it to the data, and evaluate its performance.

    Args:
    - df: DataFrame containing the input data.
    - feats: List of feature names for R² calculation.
    - model_type: The type of clustering model to use ("kmeans", "hierarchical", "dbscan", "hdbscan", "meanshift").
    - **kwargs: Additional arguments to pass to the clustering model.

    Returns:
    - Dictionary with R², silhouette score, and Calinski-Harabasz index.
    """

    # Select and create the clustering model
    if model_type == "kmeans":
        model = KMeans(**kwargs)
    elif model_type == "hierarchical":
        model = AgglomerativeClustering(**kwargs)
    elif model_type == 'dbscan':
        model = DBSCAN(**kwargs)
    elif model_type == 'hdbscan':
        model = HDBSCAN(**kwargs)
    elif model_type == 'meanshift':
        model = MeanShift(**kwargs)
    elif model_type == 'gmm':
        model = GaussianMixture(**kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}.")
    
    # Fit the model and get labels
    labels = model.fit_predict(df)
    
    # Evaluate clustering performance
    r2, silhouette, calinski_harabasz = cluster_evaluation(df, feats, labels)
    
    # Return results
    return {
        "Model": model_type,
        **kwargs,  # Include any model-specific parameters
        "R2": r2[0],
        "Silhouette": silhouette[0],
        "Calinski-Harabasz": calinski_harabasz[0]
    }

# Plot Cluster Evaluation
def plot_evaluation_scores(df, path=None):
    """
    Plots R², Silhouette, and Calinski-Harabasz scores for both KMeans and Hierarchical clustering models 
    (with different linkage methods) in one plot for each score, across different numbers of clusters.
    
    Args:
    - df: DataFrame containing clustering results with columns 
           ['Model', 'n_clusters', 'linkage', 'metric', 'R2', 'Silhouette', 'Calinski-Harabasz'].
    """
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Define the scores to plot
    scores = ['R2', 'Silhouette', 'Calinski-Harabasz']
    
    # Initialize a set for storing unique legend labels
    legend_handles = []
    
    # Iterate through each score
    for idx, score_name in enumerate(scores):
        plt.subplot(3, 1, idx+1)

        # Plot KMeans scores if available
        if 'kmeans' in df['Model'].values:
            kmeans_data = df[df['Model'] == 'kmeans']
            line, = plt.plot(
                kmeans_data['n_clusters'], 
                kmeans_data[score_name], 
                marker='o', 
                label='KMeans', 
                linewidth=2
            )
            if idx == 0:
                legend_handles.append(line)  # Add to legend once
        
        # Plot Hierarchical clustering scores for each linkage method
        if 'hierarchical' in df['Model'].values:
            df['metric'] = df['metric'].fillna('euclidean')
            hierarchical_data = df[df['Model'] == 'hierarchical']
            unique_linkages = hierarchical_data[['linkage', 'metric']].drop_duplicates()
            
            for _, row in unique_linkages.iterrows():
                linkage = row['linkage']
                metric = row['metric']
                
                subset = hierarchical_data[
                    (hierarchical_data['linkage'] == linkage) &
                    (hierarchical_data['metric'] == metric)
                ]
                line, = plt.plot(
                    subset['n_clusters'], 
                    subset[score_name], 
                    marker='o', 
                    label=f"{linkage.capitalize()} (Metric: {metric})", 
                    linewidth=2
                )
                if idx == 0:
                    legend_handles.append(line)  # Add to legend once
                 
       # Plot DBSCAN scores if available
        if 'dbscan' in df['Model'].values:
            dbscan_data = df[df['Model'] == 'dbscan']

            # Sort by eps so the line connects the dots in the correct order
            dbscan_data_sorted = dbscan_data.sort_values(by='eps')

            # Plot all algorithms in the same plot, connecting the dots with a line
            unique_algorithm = dbscan_data['algorithm'].drop_duplicates()
            
            line_styles = {'ball_tree': '-',
                           'kd_tree': '--',
                           'brute': ':'}

            for algorithm in unique_algorithm:
                subset = dbscan_data_sorted[dbscan_data_sorted['algorithm'] == algorithm]
                
                linestyle = line_styles.get(algorithm, '-')

                # Plot the line for the current algorithm
                line, = plt.plot(
                    subset['eps'],  
                    subset[score_name], 
                    marker='o', 
                    linestyle=linestyle,  
                    label=f"DBSCAN (Algorithm: {algorithm})", 
                    linewidth=2
                )
                if idx == 0:
                    legend_handles.append(line)  # Add to legend once


        # Plot HDBSCAN scores if available
        if 'hdbscan' in df['Model'].values:
            hdbscan_data = df[df['Model'] == 'hdbscan']
            unique_min_cluster_sizes = hdbscan_data['min_cluster_size'].drop_duplicates()
            unique_methods = hdbscan_data['cluster_selection_method'].drop_duplicates()

            for method in unique_methods:
                subset = hdbscan_data[hdbscan_data['cluster_selection_method'] == method]
                line, = plt.plot(
                    subset['min_cluster_size'],  
                    subset[score_name], 
                    marker='o', 
                    linestyle='-',  
                    label=f"HDBSCAN (Method: {method})", 
                    linewidth=2
                )
                if idx == 0:
                    legend_handles.append(line)  # Add to legend once
                    
        if 'gmm' in df['Model'].values:
            gmm_data = df[df['Model'] == 'gmm']
            unique_covariance_types = gmm_data['covariance_type'].unique()  # Get unique covariance types

            for covariance in unique_covariance_types:
                # Filter data by covariance type
                covariance_data = gmm_data[gmm_data['covariance_type'] == covariance]

                # Plot GMM results for the specific covariance type, using n_components for x-axis
                line, = plt.plot(
                    covariance_data['n_components'], 
                    covariance_data[score_name], 
                    marker='o', 
                    label=f'GMM (Covariance: {covariance})', 
                    linewidth=2
                )
                if idx == 0:
                    legend_handles.append(line)


        # Customize the plot
        plt.title(f"{score_name} Score", fontsize=16)
        plt.ylabel(f"{score_name} Score", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adjust x-ticks for different models
        if 'hdbscan' in df['Model'].values:
            plt.xticks(sorted(df['min_cluster_size'].unique())) 
            plt.xlabel("Minimum Cluster Size", fontsize=12)
        elif 'dbscan' in df['Model'].values:
            plt.xticks(sorted(df['eps'].unique()))
            plt.xlabel("Epsilon", fontsize=12)
        elif 'gmm' in df['Model'].values:
            plt.xticks(sorted(df['n_components'].unique()))
            plt.xlabel("N_components", fontsize=12)
        else:
            plt.xticks(range(int(min(df['n_clusters'])), int(max(df['n_clusters'])) + 1))
            plt.xlabel("Number of Clusters", fontsize=12)
    
    # Add a single legend for all subplots, located outside the last subplot
    plt.legend(
        handles=legend_handles, 
        title='Clustering Methods', 
        fontsize=10, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=2  # Arrange in 2 columns if there are many methods
    )

    # Adjust layout to prevent overlapping subplots
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for the legend
    if path != None:
        plt.savefig(f'{path}_scores.png')
    plt.show()


# Model Comparison
def model_comparison(df_melted, metrics, model_for_line, line_color='red', line_style='--', marker='o', figsize=(8, 6)):
    """
    Plot line charts for different metrics and add a vertical line at the specified model.

    Parameters:
    - df_melted: DataFrame containing the melted data.
    - metrics: List of metrics to plot (e.g., ['R2', 'Silhouette', 'Calinski-Harabasz']).
    - model_for_line: The model name where you want to add the vertical line.
    - line_color: Color for the vertical line (default is 'red').
    - line_style: Line style for the vertical line (default is dashed '--').
    - marker: Marker style for the line plot (default is 'o').
    - figsize: Figure size for the plot (default is (8, 6)).
    """
    # Iterate over each metric and create a separate plot
    for metric in metrics:
        # Filter data for the current metric
        data_metric = df_melted[df_melted['Metric'] == metric]

        # Create a new figure for each metric
        plt.figure(figsize=figsize)

        # Plot the data with a specific line color and no legend
        sns.lineplot(
            data=data_metric,
            x='Configuration',
            y='Value',
            marker=marker,
            color='#568789',  
            legend=False,
            ci=None
        )

        # Add the vertical line at the specified model
        plt.axvline(x=model_for_line, color=line_color, linestyle=line_style, label=f'{model_for_line} (Vertical Line)')

        # Remove the grid
        plt.grid(False)

        # Set title and labels
        plt.title(f'{metric} by Model')
        plt.xlabel('Model')
        plt.ylabel('Value')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

        # Adjust the layout for each individual plot
        plt.tight_layout()

        # Show the plot
        plt.show()