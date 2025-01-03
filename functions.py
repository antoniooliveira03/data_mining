# This file will consist of functions used through the notebooks

#################### Imports ##############################
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



main_color= '#568789'


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

def plot_distribution_and_boxplot(df, column_name, color=main_color):
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

#################### Feature Engineering ##############################
def avg_hour(row):
    """
    Computes the average hour when orders were placed, 
    weighted by the number of orders at each hour.
    If no orders are placed, returns 0.
    """
    total_orders = row.sum()
    
    if total_orders == 0:
        return 0  
    
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

def plot_hierarchical_dendrograms(data, path=None, linkages=["ward", "complete", "average", "single"], metrics=['euclidean']):
    """
    Create and display a grid of hierarchical clustering dendrograms for the given data,
    enumerating different linkage and metric combinations.

    Args:
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



def plot_dim_reduction(embedding, targets=None, 
                       technique='UMAP',
                       figsize=(10, 7)):

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
            color = scatter.cmap(scatter.norm(i))  # Use colormap and normalize to get the correct color
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

## K-Means Silhouette
def plot_silhouette(temp_data, possible_k):   
    avg_silhouette = []
    for k in possible_k:

        # Initialize and fit KMeans
        kmclust = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=42)
        cluster_labels = kmclust.fit_predict(temp_data)

        # Compute silhouette scores
        silhouette_avg = silhouette_score(temp_data, cluster_labels)
        avg_silhouette.append(silhouette_avg)
        print(f"For n_clusters = {k}, the average silhouette_score is: {silhouette_avg:.4f}")

        sample_silhouette_values = silhouette_samples(temp_data, cluster_labels)

        # Create figure
        fig, ax = plt.subplots(figsize=(13, 7))

        # Initialize vertical position for plotting clusters
        y_position = 0

        for i in range(k):
            # Extract and sort silhouette scores for cluster i
            cluster_values = sample_silhouette_values[cluster_labels == i]
            cluster_values.sort()

            # Compute cluster size
            cluster_size = len(cluster_values)

            # Fill the silhouette scores for cluster i
            color = cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(
                range(y_position, y_position + cluster_size),
                0,
                cluster_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label clusters in the middle of each block
            ax.text(-0.05, y_position + cluster_size / 2, str(i))

            # Update y_position for the next cluster
            y_position += cluster_size + 10

        # Plot the average silhouette score as a vertical line
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Avg: {silhouette_avg:.2f}")

        # Customize axes
        ax.set_title(f"Silhouette Plot for k = {k}", fontsize=16)
        ax.set_xlabel("Silhouette Coefficient", fontsize=12)
        ax.set_ylabel("Cluster Label", fontsize=12)
        ax.set_xlim([-0.1, 1.0])
        ax.set_ylim([0, y_position])
        ax.legend(loc="upper right")

        # Display the plot
        plt.show()

    return avg_silhouette


## SOM Hexagons
def plot_som_hexagons(som,
                      matrix,
                      cmap=cm.Blues,
                      figsize=(20,20),
                      annotate=True,
                      title="SOM Matrix",
                      cbar_label="Color Scale"
                ):

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


## Cluster Profiling
def plot_cluster_profiling(df, cluster_labels, cluster_method_name, 
                           figsize=(6, 8), cmap="BrBG", fmt=".2f"):
    
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

## R2

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


def get_ssb(df, feats, label_col):
    
    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))) )

    ssb = np.sum(ssb_i)
    

    return ssb


def get_ssw(df, feats, label_col):

    feats_label = feats+[label_col]

    df_k = df[feats_label].groupby(by=label_col).apply(lambda col: get_ss(col, feats), 
                                                       include_groups=False)

    return df_k.sum()

def get_rsq(df, feats, label_col):
    df_sst_ = get_ss(df, feats)                 # get total sum of squares
    df_ssw_ = get_ssw(df, feats, label_col)     # get ss within
    df_ssb_ = df_sst_ - df_ssw_                 # get ss between

    # r2 = ssb/sst 
    return (df_ssb_/df_sst_)
    
def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):

    r2 = []  # where we will store the R2 metrics for each cluster solution
    feats = df.columns.tolist()
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)
        
        #get cluster labels
        hclabels = cluster.fit_predict(df) 
        
        # concat df with labels
        df_concat = pd.concat([df, pd.Series(hclabels, name='labels', index=df.index)], axis=1)  
        
        
        # append the R2 of the given cluster solution
        r2.append(get_rsq(df_concat, feats, 'labels'))

        
    return np.array(r2)

def cluster_evaluation(df, feats, labels):

    r2 = []  # where we will store the R2 metrics for each cluster solution
    silhouette = []
    calinski_harabasz = []

    # concat df with labels
    df_concat = pd.concat([df, pd.Series(labels, name='labels', index=df.index)], axis=1)   

    # append the R2 of the given cluster solution
    r2.append(get_rsq(df_concat, feats, 'labels'))
    # append silhouette score
    silhouette.append(silhouette_score(df, labels))
    # append calinski_harabasz score
    calinski_harabasz.append(calinski_harabasz_score(df, labels))

    return np.array(r2), np.array(silhouette), np.array(calinski_harabasz)

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
    elif model_type == 'gaussian':
        model = GaussianMixture(**kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'kmeans' or 'hierarchical'.")
    
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

        # Plot GMM scores if available
        if 'gaussian' in df['Model'].values:
            gaussian_data = df[df['Model'] == 'gaussian']
            unique_n_components = gaussian_data['n_components'].drop_duplicates()

            for n in unique_n_components:
                subset = gaussian_data[gaussian_data['n_components'] == n]
                line, = plt.plot(
                    subset['n_components'],  
                    subset[score_name], 
                    marker='o', 
                    label=f"GMM (n_components={n})", 
                    linewidth=2
                )
                if idx == 0:
                    legend_handles.append(line)  # Add to legend once

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