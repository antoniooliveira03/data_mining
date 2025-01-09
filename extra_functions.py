import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import sklearn.manifold
import numpy as np
import plotly_express as px
from streamlit_option_menu import option_menu

# Define the function
def cluster_profiles(df, label_columns, figsize, cmap="tab10", compare_titles=None):
    """
    Pass df with labels columns of one or multiple clustering labels. 
    Then specify these label columns to perform the cluster profile according to them.
    """
    if compare_titles is None:
        compare_titles = [""] * len(label_columns)
        
    fig, axes = plt.subplots(nrows=len(label_columns), 
                            ncols=1,
                            figsize=figsize, 
                            constrained_layout=True,
                            squeeze=False)
    for ax, label, titl in zip(axes, label_columns, compare_titles):
        # Filtering df
        drop_cols = [i for i in label_columns if i != label]
        dfax = df.drop(drop_cols, axis=1)
        
        # Getting the cluster centroids
        centroids = dfax.groupby(by=label, as_index=False).mean()
        
        # Convert label column to string for plotting
        centroids[label] = centroids[label].astype(str)
        
        # Setting Data
        pd.plotting.parallel_coordinates(
            centroids, 
            label, 
            color=sns.color_palette(cmap),
            ax=ax[0]
        )

        # Remove vertical gridlines
        ax[0].grid(False)
        
        # Hide vertical lines by overlaying a white box
        for spine in ["top", "right", "bottom", "left"]:
            ax[0].spines[spine].set_visible(False)
        
        # Customize layout
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
        ax[0].annotate(text=titl, xy=(0.95, 1.1), xycoords='axes fraction', fontsize=13, fontweight='heavy') 
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=13)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), 
                            rotation=40,
                            ha='right')
        
        ax[0].legend(handles, cluster_labels,
                    loc='center left', bbox_to_anchor=(1, 0.5), title=label)
        
    plt.suptitle("Cluster Simple Profiling", fontsize=23)
    st.pyplot(fig)

def plot_cluster_profiles(data, features, label_column):
    """Plot cluster profiles."""
    st.write(f"Plotting Cluster Profiles for Selected Segment")
    cluster_profiles(
        df=data[features + [label_column]], 
        label_columns=[label_column], 
        figsize=(10, 6)
    )

def plot_dim_reduction(data, technique='UMAP', n_neighbors=15, targets=None, figsize=(10, 7)):
    """
    Plots a 2D representation of high-dimensional data using UMAP or t-SNE.

    Parameters:
    - data (array-like or DataFrame): High-dimensional data to reduce.
    - technique (str): Dimensionality reduction technique name ('UMAP' or 't-SNE').
    - n_neighbors (int): Number of neighbors for UMAP (ignored for t-SNE).
    - targets (array-like): Cluster labels for data points (optional).
    - figsize (tuple): Figure size (default: (10, 7)).
    """
    plt.figure(figsize=figsize)

    # Apply UMAP or t-SNE
    if technique == 'UMAP':
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(data)
    elif technique == 't-SNE':
        reducer = sklearn.manifold.TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
    else:
        raise ValueError(f"Unsupported technique: {technique}. Choose 'UMAP' or 't-SNE'.")

    # Plotting the results
    if targets is not None:
        scatter = plt.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=np.array(targets).astype(int), 
            cmap='tab10'
        )
        
        # Create a legend for clusters
        labels = np.unique(targets)
        handles = []

        for i, label in enumerate(labels):
            color = scatter.cmap(scatter.norm(i))  
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Cluster {label}'))

        plt.legend(handles=handles, title='Clusters')

    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5)

    # Title for the plot based on the dimensionality reduction technique
    if technique == 'UMAP':
        plt.title('UMAP Projection')
    elif technique == 't-SNE':
        plt.title('t-SNE Projection')

    st.pyplot(plt)
    plt.close() 

def plot_dimensionality_reduction(data, technique, label_column, n_neighbors=None):
    """Perform and plot dimensionality reduction."""
    st.write(f"Performing Dimensionality Reduction using {technique}")
    if technique == 'UMAP':
        plot_dim_reduction(data, technique=technique, n_neighbors=n_neighbors, targets=data[label_column])
    else:
        plot_dim_reduction(data, technique=technique, targets=data[label_column])


def get_selectable_columns(columns, excluded_labels=None, excluded_categories=None):
    """Filter out label and categorical columns."""
    if excluded_labels is None:
        excluded_labels = []  
    if excluded_categories is None:
        excluded_categories = []  
    
    # Exclude both labels and categories
    return [col for col in columns if col not in excluded_labels and col not in excluded_categories]


# Defining and calling the function for an interactive bar chart
def interactive_bar(dataframe, binary_features, categorical_features):
    cat_feature = st.selectbox('Feature', options=categorical_features)
    color_choice = st.color_picker('Select a plot colour', '#1f77b7')

    # If the selected feature is 'region', reclassify it to make it more readable
    if cat_feature == 'customer_region': 
        dataframe[cat_feature] = dataframe[cat_feature].apply(lambda x: f"Region {x}")
    
    # Loop through the list of binary features and apply the map operation
    for feature in binary_features:
        if feature in dataframe.columns:
            dataframe[feature] = dataframe[feature].map({0: 'No', 1: 'Yes'})

    # Create the histogram
    bar_chart = px.histogram(dataframe, x=cat_feature)

    # Update the color of the bars using the color choice
    bar_chart.update_traces(marker=dict(color=color_choice))

    # Add additional customization, like titles and axis labels
    bar_chart.update_layout(
        xaxis_title=cat_feature,
        yaxis_title='Count',
        title=f"Distribution of {cat_feature}",
        template="plotly_dark"  
    )

    # Display the chart
    st.plotly_chart(bar_chart)


# Defining and calling the function for an interactive scatterplot
def interactive_scater (dataframe, numeric_features):
    x_axis_val = st.selectbox('Select X-Axis Value', options=numeric_features)
    y_axis_val = st.selectbox('Select Y-Axis Value', options=numeric_features)
    col = st.color_picker('Select a plot colour')

    plot  = px.scatter(dataframe, x=x_axis_val, y=y_axis_val)
    plot.update_traces(marker = dict(color=col))
    st.plotly_chart(plot)


def interactive_hist (dataframe, numeric_features):
    box_hist = st.selectbox('Feature', options=numeric_features)
    color_choice = st.color_picker('Select a plot colour', '#1f77b4')
    bin_count = st.slider('Select number of bins', min_value=5, max_value=100, value=20, step=1)

    hist  = sns.displot(dataframe[box_hist], color=color_choice, bins=bin_count)
    
    plt.title(f"Histogram of {box_hist}")
    st.pyplot(hist)

     
def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Explore Data", "Clustering", "About Us"],
        icons=["house", "bar-chart-line", 'basket', "person"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    return selected


def plot_boxplot_by_cluster_streamlit(df, labels_col, y_col):
    """
    Plot a box plot of a given column against cluster labels with different colors in Streamlit.
    
    Parameters:
    - df: DataFrame containing the data
    - labels_col: Column name for cluster labels (clusters)
    - y_col: Column name for the y-axis (the variable to compare by clusters)
    """
    fig = px.box(df, x=labels_col, y=y_col, color=labels_col,  
                 labels={labels_col: 'Cluster', y_col: y_col},
                 title=f'Box Plot of {y_col} by Cluster')

    fig.update_traces(marker=dict(size=8))  
    
    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
